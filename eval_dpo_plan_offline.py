from pathlib import Path

import hydra
import torch
import stable_worldmodel as swm
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from train_dpo import DPOPairsDataset
from train_dpo_plan import GaussianPlanPolicyNoAct
from make_dpo_pairs import rollout_cost_from_latents, freeze_model


def get_cfg(cfg: DictConfig):
    default = OmegaConf.create(
        {
            "pairs": "dpo/cube_pairs_eval_h5_dataset_5k.pt",
            "policy_dir": "dpo_plan_eval_h5_dataset_5k_bc005",
            "policy_file": "dpo_policy.pt",
            "model_dir": "cube",
            "model_file": "lewm_object.ckpt",
            "batch_size": 256,
            "num_samples": 32,
            "device": "cuda",
        }
    )
    if "dpo_eval" in cfg:
        return OmegaConf.merge(default, cfg.dpo_eval)
    return default


def load_plan_policy(policy_path: Path, device: str):
    payload = torch.load(policy_path, map_location="cpu")

    policy_type = payload.get("policy_type", None)
    if policy_type != "GaussianPlanPolicyNoAct":
        print(f"[Warning] Expected policy_type='GaussianPlanPolicyNoAct', got {policy_type}")

    pcfg = payload["policy_config"]

    policy = GaussianPlanPolicyNoAct(
        ctx_len=int(pcfg["ctx_len"]),
        embed_dim=int(pcfg["embed_dim"]),
        action_dim=int(pcfg["action_dim"]),
        horizon=int(pcfg["horizon"]),
        hidden_dim=int(pcfg.get("hidden_dim", 512)),
        depth=int(pcfg.get("depth", 3)),
        use_layernorm=bool(pcfg.get("use_layernorm", True)),
        log_std_min=float(pcfg.get("log_std_min", -5.0)),
        log_std_max=float(pcfg.get("log_std_max", 2.0)),
    )

    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    policy = policy.to(device).eval()

    return policy, payload


@torch.no_grad()
def main_eval(cfg: DictConfig):
    eval_cfg = get_cfg(cfg)

    device = str(eval_cfg.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cache_dir = Path(swm.data.utils.get_cache_dir())

    pairs_path = cache_dir / str(eval_cfg.pairs)
    policy_path = cache_dir / str(eval_cfg.policy_dir) / str(eval_cfg.policy_file)
    model_path = cache_dir / str(eval_cfg.model_dir) / str(eval_cfg.model_file)

    print("pairs_path :", pairs_path)
    print("policy_path:", policy_path)
    print("model_path :", model_path)

    if not pairs_path.exists():
        raise FileNotFoundError(f"Cannot find pairs file: {pairs_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"Cannot find policy file: {policy_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find LeWM model file: {model_path}")

    dataset = DPOPairsDataset(pairs_path)

    print("\n==== Pair dataset meta ====")
    print(dataset.meta)
    print("num_pairs:", len(dataset))
    print("z_ctx shape :", tuple(dataset.z_ctx.shape))
    print("z_goal shape:", tuple(dataset.z_goal.shape))
    print("a_ctx shape :", tuple(dataset.a_ctx.shape))
    print("a_pos shape :", tuple(dataset.a_pos.shape))
    print("a_neg shape :", tuple(dataset.a_neg.shape))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(eval_cfg.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
    )

    policy, payload = load_plan_policy(policy_path, device=device)

    print("\n==== Policy config ====")
    print(payload.get("policy_type", None))
    print(payload.get("policy_config", {}))
    print(payload.get("dpo_train_config", {}))

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model = freeze_model(model).to(device)

    total = 0

    dpo_acc_sum = 0.0
    logp_gap_sum = 0.0

    log_std_sum = 0.0

    cost_pos_sum = 0.0
    cost_neg_sum = 0.0

    cost_policy_mean_sum = 0.0

    cost_sample_mean_sum = 0.0
    cost_sample_best_sum = 0.0
    cost_sample_std_sum = 0.0

    cost_random_mean_sum = 0.0
    cost_random_best_sum = 0.0
    cost_random_std_sum = 0.0

    policy_best_better_than_random_best_sum = 0.0
    policy_mean_better_than_random_mean_sum = 0.0
    policy_mean_better_than_pos_sum = 0.0
    policy_best_better_than_pos_sum = 0.0

    for batch in tqdm(loader, desc="Offline DPO planner eval"):
        batch = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        z_ctx = batch["z_ctx"]
        z_goal = batch["z_goal"]
        a_ctx = batch["a_ctx"]
        a_pos = batch["a_pos"]
        a_neg = batch["a_neg"]

        B = z_ctx.shape[0]
        total += B

        # ------------------------------------------------------------
        # 1. Preference classification
        # ------------------------------------------------------------
        logp_pos = policy.log_prob(z_ctx, z_goal, a_pos)
        logp_neg = policy.log_prob(z_ctx, z_goal, a_neg)

        dpo_acc_sum += (logp_pos > logp_neg).float().sum().item()
        logp_gap_sum += (logp_pos - logp_neg).sum().item()

        # ------------------------------------------------------------
        # 2. Stored positive / negative costs
        # ------------------------------------------------------------
        cost_pos = rollout_cost_from_latents(
            model=model,
            z_ctx=z_ctx,
            a_ctx=a_ctx,
            future_actions=a_pos[:, None],
            z_goal=z_goal,
            history_size=z_ctx.shape[1],
        ).squeeze(1)

        cost_neg = rollout_cost_from_latents(
            model=model,
            z_ctx=z_ctx,
            a_ctx=a_ctx,
            future_actions=a_neg[:, None],
            z_goal=z_goal,
            history_size=z_ctx.shape[1],
        ).squeeze(1)

        cost_pos_sum += cost_pos.sum().item()
        cost_neg_sum += cost_neg.sum().item()

        # ------------------------------------------------------------
        # 3. Deterministic mean action cost
        # ------------------------------------------------------------
        mean_action, log_std = policy(z_ctx, z_goal)
        mean_action = mean_action[:, None]  # (B, 1, H, A)

        cost_policy_mean = rollout_cost_from_latents(
            model=model,
            z_ctx=z_ctx,
            a_ctx=a_ctx,
            future_actions=mean_action,
            z_goal=z_goal,
            history_size=z_ctx.shape[1],
        ).squeeze(1)

        cost_policy_mean_sum += cost_policy_mean.sum().item()
        log_std_sum += log_std.mean().item() * B

        # ------------------------------------------------------------
        # 4. Stochastic policy samples
        # ------------------------------------------------------------
        sampled, _ = policy.sample(
            z_ctx=z_ctx,
            z_goal=z_goal,
            num_samples=int(eval_cfg.num_samples),
        )

        cost_sample = rollout_cost_from_latents(
            model=model,
            z_ctx=z_ctx,
            a_ctx=a_ctx,
            future_actions=sampled,
            z_goal=z_goal,
            history_size=z_ctx.shape[1],
        )

        sample_mean = cost_sample.mean(dim=1)
        sample_best = cost_sample.min(dim=1).values
        sample_std = cost_sample.std(dim=1)

        cost_sample_mean_sum += sample_mean.sum().item()
        cost_sample_best_sum += sample_best.sum().item()
        cost_sample_std_sum += sample_std.sum().item()

        # ------------------------------------------------------------
        # 5. Random normalized action baseline
        # ------------------------------------------------------------
        _, S, H, A = sampled.shape
        random_actions = torch.randn(B, S, H, A, device=device)

        cost_random = rollout_cost_from_latents(
            model=model,
            z_ctx=z_ctx,
            a_ctx=a_ctx,
            future_actions=random_actions,
            z_goal=z_goal,
            history_size=z_ctx.shape[1],
        )

        random_mean = cost_random.mean(dim=1)
        random_best = cost_random.min(dim=1).values
        random_std = cost_random.std(dim=1)

        cost_random_mean_sum += random_mean.sum().item()
        cost_random_best_sum += random_best.sum().item()
        cost_random_std_sum += random_std.sum().item()

        # ------------------------------------------------------------
        # 6. Diagnostics
        # ------------------------------------------------------------
        policy_best_better_than_random_best_sum += (sample_best < random_best).float().sum().item()
        policy_mean_better_than_random_mean_sum += (sample_mean < random_mean).float().sum().item()

        policy_mean_better_than_pos_sum += (cost_policy_mean < cost_pos).float().sum().item()
        policy_best_better_than_pos_sum += (sample_best < cost_pos).float().sum().item()

    print("\n==== Offline DPO Planner Eval ====")
    print(f"N: {total}")

    print("\n[Preference classification]")
    print(f"logp acc pos>neg                 : {dpo_acc_sum / total:.4f}")
    print(f"mean logp gap                    : {logp_gap_sum / total:.4f}")

    print("\n[Stored pair costs]")
    print(f"mean cost pos                    : {cost_pos_sum / total:.4f}")
    print(f"mean cost neg                    : {cost_neg_sum / total:.4f}")
    print(f"mean cost gap neg-pos            : {(cost_neg_sum - cost_pos_sum) / total:.4f}")

    print("\n[Policy deterministic mean]")
    print(f"policy mean action cost          : {cost_policy_mean_sum / total:.4f}")
    print(f"policy mean log_std              : {log_std_sum / total:.4f}")
    print(f"policy mean better than pos rate : {policy_mean_better_than_pos_sum / total:.4f}")

    print("\n[Policy stochastic samples]")
    print(f"policy sample mean cost          : {cost_sample_mean_sum / total:.4f}")
    print(f"policy sample best cost          : {cost_sample_best_sum / total:.4f}")
    print(f"policy sample cost std           : {cost_sample_std_sum / total:.4f}")
    print(f"policy best better than pos rate : {policy_best_better_than_pos_sum / total:.4f}")

    print("\n[Random normalized samples]")
    print(f"random sample mean cost          : {cost_random_mean_sum / total:.4f}")
    print(f"random sample best cost          : {cost_random_best_sum / total:.4f}")
    print(f"random sample cost std           : {cost_random_std_sum / total:.4f}")

    print("\n[Policy vs random]")
    print(f"policy mean < random mean rate   : {policy_mean_better_than_random_mean_sum / total:.4f}")
    print(f"policy best < random best rate   : {policy_best_better_than_random_best_sum / total:.4f}")


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg: DictConfig):
    main_eval(cfg)


if __name__ == "__main__":
    run()