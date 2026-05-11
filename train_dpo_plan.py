import copy
import json
import time
from pathlib import Path
from typing import Dict

import hydra
import lightning as pl
import stable_worldmodel as swm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict

from train_dpo import DPOPairsDataset


class GaussianPlanPolicyNoAct(nn.Module):
    """
    DPO planner without action-context leakage.

    Input:
        z_ctx:  (B, C, D)
        z_goal: (B, D)

    Output:
        Gaussian distribution over future action sequence:
        actions: (B, H, A)
    """

    def __init__(
        self,
        ctx_len: int,
        embed_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int = 512,
        depth: int = 3,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.ctx_len = ctx_len
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.use_layernorm = use_layernorm

        input_dim = ctx_len * embed_dim + embed_dim

        layers = []
        if use_layernorm:
            layers.append(nn.LayerNorm(input_dim))

        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, horizon * action_dim)
        self.log_std_head = nn.Linear(hidden_dim, horizon * action_dim)

        nn.init.zeros_(self.mean_head.bias)
        nn.init.constant_(self.log_std_head.bias, -0.5)

    def _features(self, z_ctx: torch.Tensor, z_goal: torch.Tensor):
        B = z_ctx.shape[0]
        z_flat = z_ctx.reshape(B, -1)
        x = torch.cat([z_flat, z_goal], dim=-1)
        return x

    def forward(self, z_ctx: torch.Tensor, z_goal: torch.Tensor):
        x = self._features(z_ctx, z_goal)
        h = self.backbone(x)

        mean = self.mean_head(h).view(-1, self.horizon, self.action_dim)
        log_std = self.log_std_head(h).view(-1, self.horizon, self.action_dim)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def log_prob(self, z_ctx: torch.Tensor, z_goal: torch.Tensor, actions: torch.Tensor):
        mean, log_std = self.forward(z_ctx, z_goal)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(dim=(-1, -2))

    @torch.no_grad()
    def sample(self, z_ctx: torch.Tensor, z_goal: torch.Tensor, num_samples: int = 1):
        mean, log_std = self.forward(z_ctx, z_goal)
        std = log_std.exp()

        B, H, A = mean.shape
        mean = mean[:, None].expand(B, num_samples, H, A)
        std = std[:, None].expand(B, num_samples, H, A)

        dist = torch.distributions.Normal(mean, std)
        actions = dist.sample()
        logp = dist.log_prob(actions).sum(dim=(-1, -2))

        return actions, logp


def dpo_loss(
    policy_logp_pos: torch.Tensor,
    policy_logp_neg: torch.Tensor,
    ref_logp_pos: torch.Tensor,
    ref_logp_neg: torch.Tensor,
    beta: float,
):
    policy_log_ratio = policy_logp_pos - policy_logp_neg
    ref_log_ratio = ref_logp_pos - ref_logp_neg

    logits = beta * (policy_log_ratio - ref_log_ratio)
    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        acc = (logits > 0).float().mean()
        margin = (policy_log_ratio - ref_log_ratio).mean()
        chosen_reward = beta * (policy_logp_pos - ref_logp_pos)
        rejected_reward = beta * (policy_logp_neg - ref_logp_neg)

    metrics = {
        "dpo_loss": loss.detach(),
        "dpo_acc": acc.detach(),
        "dpo_margin": margin.detach(),
        "chosen_reward": chosen_reward.mean().detach(),
        "rejected_reward": rejected_reward.mean().detach(),
        "reward_gap": (chosen_reward - rejected_reward).mean().detach(),
    }

    return loss, metrics


class DPOPlannerModule(pl.LightningModule):
    def __init__(
        self,
        policy: GaussianPlanPolicyNoAct,
        ref_policy: GaussianPlanPolicyNoAct,
        cfg: DictConfig,
    ):
        super().__init__()

        self.policy = policy
        self.ref_policy = ref_policy
        self.cfg_obj = cfg

        self.beta = float(cfg.dpo_train.beta)
        self.lr = float(cfg.dpo_train.lr)
        self.weight_decay = float(cfg.dpo_train.weight_decay)
        self.bc_weight = float(cfg.dpo_train.bc_weight)

        self.ref_policy.eval()
        self.ref_policy.requires_grad_(False)

    def _step(self, batch: Dict[str, torch.Tensor], stage: str):
        z_ctx = batch["z_ctx"]
        z_goal = batch["z_goal"]
        a_pos = batch["a_pos"]
        a_neg = batch["a_neg"]

        policy_logp_pos = self.policy.log_prob(z_ctx, z_goal, a_pos)
        policy_logp_neg = self.policy.log_prob(z_ctx, z_goal, a_neg)

        with torch.no_grad():
            ref_logp_pos = self.ref_policy.log_prob(z_ctx, z_goal, a_pos)
            ref_logp_neg = self.ref_policy.log_prob(z_ctx, z_goal, a_neg)

        loss, metrics = dpo_loss(
            policy_logp_pos=policy_logp_pos,
            policy_logp_neg=policy_logp_neg,
            ref_logp_pos=ref_logp_pos,
            ref_logp_neg=ref_logp_neg,
            beta=self.beta,
        )

        bc_loss = -policy_logp_pos.mean()
        total_loss = loss + self.bc_weight * bc_loss

        with torch.no_grad():
            logs = {
                f"{stage}/loss": total_loss.detach(),
                f"{stage}/dpo_loss": metrics["dpo_loss"],
                f"{stage}/bc_loss": bc_loss.detach(),
                f"{stage}/dpo_acc": metrics["dpo_acc"],
                f"{stage}/dpo_margin": metrics["dpo_margin"],
                f"{stage}/chosen_reward": metrics["chosen_reward"],
                f"{stage}/rejected_reward": metrics["rejected_reward"],
                f"{stage}/reward_gap": metrics["reward_gap"],
                f"{stage}/policy_logp_pos": policy_logp_pos.mean().detach(),
                f"{stage}/policy_logp_neg": policy_logp_neg.mean().detach(),
                f"{stage}/ref_logp_pos": ref_logp_pos.mean().detach(),
                f"{stage}/ref_logp_neg": ref_logp_neg.mean().detach(),
            }

            if "cost_pos" in batch:
                logs[f"{stage}/cost_pos"] = batch["cost_pos"].float().mean().detach()
            if "cost_neg" in batch:
                logs[f"{stage}/cost_neg"] = batch["cost_neg"].float().mean().detach()
            if "cost_pos" in batch and "cost_neg" in batch:
                logs[f"{stage}/cost_gap"] = (
                    batch["cost_neg"].float() - batch["cost_pos"].float()
                ).mean().detach()

        self.log_dict(
            logs,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


# class SavePolicyCallback(pl.Callback):
#     def __init__(self, run_dir: Path, filename: str = "dpo_policy.pt"):
#         super().__init__()
#         self.run_dir = Path(run_dir)
#         self.filename = filename

class SavePolicyCallback(pl.Callback):
    def __init__(self, run_dir: Path, filename: str = "dpo_policy.pt", train_start_time: float | None = None,):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.filename = filename
        self.train_start_time = train_start_time

    def _save(self, trainer: pl.Trainer, pl_module: DPOPlannerModule, tag: str):
        if not trainer.is_global_zero:
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)

        path = self.run_dir / self.filename
        training_time_sec = None
        if self.train_start_time is not None:
            training_time_sec = time.perf_counter() - self.train_start_time

        payload = {
            "policy_type": "GaussianPlanPolicyNoAct",
            "policy_state_dict": pl_module.policy.state_dict(),
            "policy_config": {
                "ctx_len": pl_module.policy.ctx_len,
                "embed_dim": pl_module.policy.embed_dim,
                "action_dim": pl_module.policy.action_dim,
                "horizon": pl_module.policy.horizon,
                "hidden_dim": pl_module.policy.hidden_dim,
                "depth": pl_module.policy.depth,
                "use_layernorm": pl_module.policy.use_layernorm,
                "log_std_min": pl_module.policy.log_std_min,
                "log_std_max": pl_module.policy.log_std_max,
            },
            "dpo_train_config": OmegaConf.to_container(pl_module.cfg_obj.dpo_train, resolve=True),
            "tag": tag,
            "training_time_sec": None if training_time_sec is None else float(training_time_sec),
            "training_time_min": None if training_time_sec is None else float(training_time_sec / 60.0),
        }
        torch.save(payload, path)
        print(f"[SavePolicyCallback] saved policy to {path}")

    def on_train_epoch_end(self, trainer, pl_module):
        self._save(trainer, pl_module, tag=f"epoch_{trainer.current_epoch}")

    def on_fit_end(self, trainer, pl_module):
        self._save(trainer, pl_module, tag="final")


def get_dpo_train_cfg(cfg: DictConfig) -> DictConfig:
    default = OmegaConf.create(
        {
            "pairs": "dpo/cube_pairs_eval_h5_dataset_5k.pt",
            "subdir": "dpo_plan_eval_h5_dataset_5k_bc005",

            "seed": 42,
            "train_split": 0.9,
            "batch_size": 256,
            "num_workers": 4,
            "max_epochs": 20,
            "devices": 1,
            "accelerator": "gpu",
            "precision": "bf16-mixed",

            "beta": 0.1,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "bc_weight": 0.05,

            "hidden_dim": 512,
            "depth": 3,
            "use_layernorm": True,
            "log_std_min": -5.0,
            "log_std_max": 2.0,

            "wandb_enabled": False,
            "output_policy_name": "dpo_policy.pt",
        }
    )

    if "dpo_train" in cfg:
        return OmegaConf.merge(default, cfg.dpo_train)

    return default


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg: DictConfig):
    script_start_time = time.perf_counter()
    dpo_cfg = get_dpo_train_cfg(cfg)

    with open_dict(cfg):
        cfg.dpo_train = dpo_cfg

    torch.manual_seed(int(dpo_cfg.seed))

    pairs_path = Path(swm.data.utils.get_cache_dir(), str(dpo_cfg.pairs))
    if not pairs_path.exists():
        raise FileNotFoundError(f"Cannot find DPO pairs file: {pairs_path}")

    print(f"Loading DPO pairs from: {pairs_path}")
    dataset = DPOPairsDataset(pairs_path)

    print("==== DPO pairs meta ====")
    print(dataset.meta)

    ctx_len = int(dataset.z_ctx.shape[1])
    embed_dim = int(dataset.z_ctx.shape[2])
    action_dim = int(dataset.a_pos.shape[-1])
    horizon = int(dataset.a_pos.shape[1])

    print("==== inferred shapes ====")
    print(f"ctx_len={ctx_len}, embed_dim={embed_dim}, action_dim={action_dim}, horizon={horizon}")
    print(f"num_pairs={len(dataset)}")

    generator = torch.Generator().manual_seed(int(dpo_cfg.seed))

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        lengths=[float(dpo_cfg.train_split), 1.0 - float(dpo_cfg.train_split)],
        generator=generator,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=int(dpo_cfg.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(dpo_cfg.num_workers),
        persistent_workers=int(dpo_cfg.num_workers) > 0,
        pin_memory=True,
        generator=generator,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=int(dpo_cfg.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(dpo_cfg.num_workers),
        persistent_workers=int(dpo_cfg.num_workers) > 0,
        pin_memory=True,
    )

    policy = GaussianPlanPolicyNoAct(
        ctx_len=ctx_len,
        embed_dim=embed_dim,
        action_dim=action_dim,
        horizon=horizon,
        hidden_dim=int(dpo_cfg.hidden_dim),
        depth=int(dpo_cfg.depth),
        log_std_min=float(dpo_cfg.log_std_min),
        log_std_max=float(dpo_cfg.log_std_max),
        use_layernorm=bool(dpo_cfg.use_layernorm),
    )

    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    ref_policy.requires_grad_(False)

    module = DPOPlannerModule(policy=policy, ref_policy=ref_policy, cfg=cfg)

    run_dir = Path(swm.data.utils.get_cache_dir(), str(dpo_cfg.subdir))
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger = None
    if bool(dpo_cfg.wandb_enabled):
        logger = WandbLogger(
            project="lewm-dpo",
            name=str(dpo_cfg.subdir),
            id=str(dpo_cfg.subdir),
            resume="allow",
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="dpo-plan-{epoch:03d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=2,
        save_last=True,
    )

    train_start_time = time.perf_counter()
    
    save_policy_callback = SavePolicyCallback(
        run_dir=run_dir,
        filename=str(dpo_cfg.output_policy_name),
        train_start_time=train_start_time,
    )

    trainer = pl.Trainer(
        max_epochs=int(dpo_cfg.max_epochs),
        devices=dpo_cfg.devices,
        accelerator=str(dpo_cfg.accelerator),
        precision=dpo_cfg.precision,
        callbacks=[checkpoint_callback, save_policy_callback],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        num_sanity_val_steps=1,
        enable_checkpointing=True,
    )

    trainer.fit(module, train_loader, val_loader)

    training_time_sec = time.perf_counter() - train_start_time
    total_script_time_sec = time.perf_counter() - script_start_time
    
    if trainer.is_global_zero:
        timing = {
            "stage": "train_dpo_planner",
            "pairs_path": str(pairs_path),
            "run_dir": str(run_dir),
            "policy_path": str(run_dir / str(dpo_cfg.output_policy_name)),
            "num_pairs": int(len(dataset)),
            "train_split": float(dpo_cfg.train_split),
            "batch_size": int(dpo_cfg.batch_size),
            "max_epochs": int(dpo_cfg.max_epochs),
            "devices": dpo_cfg.devices,
            "accelerator": str(dpo_cfg.accelerator),
            "precision": str(dpo_cfg.precision),
            "beta": float(dpo_cfg.beta),
            "bc_weight": float(dpo_cfg.bc_weight),
            "training_time_sec": float(training_time_sec),
            "training_time_min": float(training_time_sec / 60.0),
            "total_script_time_sec": float(total_script_time_sec),
            "total_script_time_min": float(total_script_time_sec / 60.0),
        }

        timing_path = run_dir / "training_time.json"
        with open(timing_path, "w") as f:
            json.dump(timing, f, indent=2)

        print(f"\nTraining done. Run dir: {run_dir}")
        print(f"Policy saved to: {run_dir / str(dpo_cfg.output_policy_name)}")
        print(f"Timing saved to: {timing_path}")
        print("\nTiming:")
        print(f"  training_time_sec    : {training_time_sec:.4f}")
        print(f"  training_time_min    : {training_time_sec / 60.0:.4f}")
        print(f"  total_script_time_sec: {total_script_time_sec:.4f}")


if __name__ == "__main__":
    run()