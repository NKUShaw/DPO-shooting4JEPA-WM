from pathlib import Path
from typing import Dict
import time

import hydra
import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from utils import get_column_normalizer, get_img_preprocessor


# Example:
# export STABLEWM_HOME=/WorldModels/le-wm/datasets
#
# python make_dpo_pairs_evalstyle.py data=ogb \
#   +dpo.model_dir=cube \
#   +dpo.output=dpo/cube_pairs_eval_h5_dataset_5k.pt \
#   +dpo.num_pairs=5000 \
#   +dpo.num_candidates=16 \
#   +dpo.ctx_len=1 \
#   +dpo.plan_horizon=5 \
#   +dpo.batch_size=64 \
#   +dpo.pos_strategy=dataset \
#   +dpo.neg_strategy=global_worst


def get_dpo_cfg(cfg: DictConfig) -> DictConfig:
    default = OmegaConf.create(
        {
            # Frozen LeWM object:
            # ${STABLEWM_HOME}/${model_dir}/${model_file}
            "model_dir": "cube",
            "model_file": "lewm_object.ckpt",

            # Output under ${STABLEWM_HOME}
            "output": "dpo/cube_pairs_eval_h5_dataset.pt",

            # Pair generation
            "num_pairs": 10000,
            "num_candidates": 16,

            # Eval-style planning setup
            # cube.yaml: world.history_size=1, plan_config.horizon=5
            "ctx_len": 1,
            "plan_horizon": 5,

            # Candidate construction
            # idx 0 = dataset plan
            # idx [0, n_noisy) = dataset plan + noise
            # idx [n_noisy, S) = random N(0,1)
            "noise_std": 1.0,
            "include_dataset_action": True,

            # Preference selection
            # pos_strategy:
            #   - dataset
            #   - noisy_best
            #   - global_best
            "pos_strategy": "dataset",
            # neg_strategy:
            #   - global_worst
            #   - random
            #   - random_worse_than_pos
            "neg_strategy": "global_worst",

            # DataLoader
            "batch_size": 64,
            "num_workers": 6,

            # Runtime
            "device": "cuda",
            "save_fp16": True,
            "seed": 42,
        }
    )

    if "dpo" in cfg:
        return OmegaConf.merge(default, cfg.dpo)

    return default


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    model.requires_grad_(False)
    return model


def move_batch_to_device(batch: Dict, device: str) -> Dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def build_evalstyle_dataset_and_loader(cfg: DictConfig, dpo_cfg: DictConfig):
    """
    Build dataset with train.py-style transforms, but force num_steps to match eval-style planning.

    For plan_horizon=5:
        need pixels indices 0..5
        action plan indices 0..4
        goal frame = pixels[:, 5]
    Therefore num_steps = plan_horizon + 1.
    """
    plan_horizon = int(dpo_cfg.plan_horizon)
    required_num_steps = plan_horizon + 1

    with open_dict(cfg):
        cfg.data.dataset.num_steps = required_num_steps

    print(f"[Dataset] Overriding cfg.data.dataset.num_steps = {required_num_steps}")

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    transforms = [
        get_img_preprocessor(
            source="pixels",
            target="pixels",
            img_size=cfg.img_size,
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    dataset.transform = spt.data.transforms.Compose(*transforms)

    generator = torch.Generator().manual_seed(int(dpo_cfg.seed))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(dpo_cfg.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=int(dpo_cfg.num_workers),
        persistent_workers=int(dpo_cfg.num_workers) > 0,
        pin_memory=True,
        generator=generator,
    )

    return dataset, loader


def load_frozen_lewm(dpo_cfg: DictConfig, device: str) -> torch.nn.Module:
    model_path = Path(
        swm.data.utils.get_cache_dir(),
        str(dpo_cfg.model_dir),
        str(dpo_cfg.model_file),
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Cannot find pretrained LeWM object: {model_path}\n"
            f"Expected: $STABLEWM_HOME/{dpo_cfg.model_dir}/{dpo_cfg.model_file}"
        )

    print(f"Loading frozen LeWM from: {model_path}")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model = freeze_model(model).to(device)

    return model


@torch.no_grad()
def encode_pixels(model: torch.nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    """
    pixels: (B, T, C, H, W)
    return: (B, T, D)
    """
    out = model.encode({"pixels": pixels})
    return out["emb"]


def build_candidate_plans(
    actions: torch.Tensor,
    plan_horizon: int,
    num_candidates: int,
    noise_std: float,
    include_dataset_action: bool,
):
    """
    Args:
        actions: (B, T, A), normalized action blocks from dataset.
                 T should be plan_horizon + 1, but we only use [:plan_horizon].

    Returns:
        candidates: (B, S, plan_horizon, A)
        dataset_plan: (B, plan_horizon, A)
    """
    B, T, A = actions.shape
    device = actions.device
    dtype = actions.dtype

    if T < plan_horizon:
        raise ValueError(
            f"Need at least plan_horizon={plan_horizon} actions, got T={T}."
        )

    S = int(num_candidates)
    if S < 2:
        raise ValueError(f"num_candidates must be >= 2, got {S}")

    dataset_plan = actions[:, :plan_horizon]  # (B, H, A)

    candidates = torch.randn(B, S, plan_horizon, A, device=device, dtype=dtype)

    n_noisy = S // 2
    if n_noisy > 0:
        noisy = dataset_plan[:, None] + float(noise_std) * torch.randn(
            B,
            n_noisy,
            plan_horizon,
            A,
            device=device,
            dtype=dtype,
        )
        candidates[:, :n_noisy] = noisy

    if include_dataset_action:
        candidates[:, 0] = dataset_plan

    return candidates, dataset_plan


@torch.no_grad()
def cost_with_model_get_cost(
    model: torch.nn.Module,
    pixels_ctx: torch.Tensor,
    goal_pixels: torch.Tensor,
    action_candidates: torch.Tensor,
):
    """
    Use JEPA.rollout() + JEPA.criterion(), but manually provide goal_emb
    with the correct shape.

    Args:
        pixels_ctx: (B, ctx_len, C, H, W)
        goal_pixels: (B, 1, C, H, W)
        action_candidates: (B, S, T, A)

    Returns:
        cost: (B, S)
    """
    B, S = action_candidates.shape[:2]
    ctx_len = pixels_ctx.shape[1]

    # Expand context pixels over candidate dimension:
    # (B, ctx_len, C, H, W) -> (B, S, ctx_len, C, H, W)
    info = {
        "pixels": pixels_ctx[:, None]
        .expand(B, S, *pixels_ctx.shape[1:])
        .contiguous()
    }

    # Encode goal manually.
    # goal_z: (B, 1, D)
    goal_z = model.encode({"pixels": goal_pixels})["emb"]

    # criterion() expects goal_emb broadcastable to predicted_emb:
    # predicted_emb: (B, S, T_pred, D)
    # goal_emb should be: (B, S, 1, D)
    info["goal_emb"] = goal_z[:, None].expand(B, S, *goal_z.shape[1:]).contiguous()

    # Rollout candidate plans.
    info = model.rollout(info, action_candidates, history_size=ctx_len)

    # Compute terminal latent cost.
    cost = model.criterion(info)

    return cost


def select_positive_indices(
    cost: torch.Tensor,
    pos_strategy: str,
    num_candidates: int,
    include_dataset_action: bool,
) -> torch.Tensor:
    B, S = cost.shape
    device = cost.device
    n_noisy = max(1, int(num_candidates) // 2)

    if pos_strategy == "dataset":
        if not include_dataset_action:
            raise ValueError("pos_strategy='dataset' requires include_dataset_action=True")
        return torch.zeros(B, device=device, dtype=torch.long)

    if pos_strategy == "noisy_best":
        return cost[:, :n_noisy].argmin(dim=1)

    if pos_strategy == "global_best":
        return cost.argmin(dim=1)

    raise ValueError(
        f"Unknown pos_strategy={pos_strategy}. "
        f"Choose from ['dataset', 'noisy_best', 'global_best']."
    )


def select_negative_indices(
    cost: torch.Tensor,
    idx_pos: torch.Tensor,
    neg_strategy: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    B, S = cost.shape
    device = cost.device

    if neg_strategy == "global_worst":
        return cost.argmax(dim=1)

    if neg_strategy == "random":
        idx_neg = torch.randint(0, S, (B,), device=device, generator=generator)
        same = idx_neg == idx_pos
        idx_neg[same] = (idx_neg[same] + 1) % S
        return idx_neg

    if neg_strategy == "random_worse_than_pos":
        idx_neg = []
        for b in range(B):
            pos_cost = cost[b, idx_pos[b]]
            valid = torch.where(cost[b] > pos_cost)[0]
            valid = valid[valid != idx_pos[b]]

            if len(valid) == 0:
                idx_neg.append(cost[b].argmax())
            else:
                j = torch.randint(0, len(valid), (1,), device=device, generator=generator)
                idx_neg.append(valid[j.item()])

        return torch.stack(idx_neg, dim=0).long()

    raise ValueError(
        f"Unknown neg_strategy={neg_strategy}. "
        f"Choose from ['global_worst', 'random', 'random_worse_than_pos']."
    )


def append_batch(
    storage: Dict[str, list],
    z_ctx: torch.Tensor,
    z_goal: torch.Tensor,
    a_ctx: torch.Tensor,
    a_pos: torch.Tensor,
    a_neg: torch.Tensor,
    cost_pos: torch.Tensor,
    cost_neg: torch.Tensor,
    idx_pos: torch.Tensor,
    idx_neg: torch.Tensor,
    take: int,
    save_fp16: bool,
):
    z_ctx_save = z_ctx[:take].detach().cpu()
    z_goal_save = z_goal[:take].detach().cpu()

    if save_fp16:
        z_ctx_save = z_ctx_save.half()
        z_goal_save = z_goal_save.half()

    storage["z_ctx"].append(z_ctx_save)
    storage["z_goal"].append(z_goal_save)
    storage["a_ctx"].append(a_ctx[:take].detach().cpu())
    storage["a_pos"].append(a_pos[:take].detach().cpu())
    storage["a_neg"].append(a_neg[:take].detach().cpu())
    storage["cost_pos"].append(cost_pos[:take].detach().cpu())
    storage["cost_neg"].append(cost_neg[:take].detach().cpu())
    storage["idx_pos"].append(idx_pos[:take].detach().cpu())
    storage["idx_neg"].append(idx_neg[:take].detach().cpu())


def finalize_and_save(storage: Dict[str, list], meta: Dict, output_path: Path):
    pairs = {k: torch.cat(v, dim=0) for k, v in storage.items()}
    pairs["meta"] = meta

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pairs, output_path)

    print(f"\nSaved eval-style DPO pairs to: {output_path}")

    if "pair_generation_time_sec" in meta:
        print("\nTiming:")
        print(f"  pair_generation_time_sec: {meta['pair_generation_time_sec']:.4f}")
        print(f"  pair_generation_time_min: {meta['pair_generation_time_min']:.4f}")
        print(f"  time_per_pair_sec       : {meta['time_per_pair_sec']:.8f}")

    print("Shapes:")
    for k, v in pairs.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    cost_pos = pairs["cost_pos"].float()
    cost_neg = pairs["cost_neg"].float()
    gap = cost_neg - cost_pos

    print("\nCost gap:")
    print(f"  mean cost_pos: {cost_pos.mean().item():.6f}")
    print(f"  mean cost_neg: {cost_neg.mean().item():.6f}")
    print(f"  mean gap     : {gap.mean().item():.6f}")

    idx_pos = pairs["idx_pos"]
    idx_neg = pairs["idx_neg"]
    num_candidates = int(meta["num_candidates"])

    print("\nIndex stats:")
    print(f"  pos is dataset action ratio: {(idx_pos == 0).float().mean().item():.6f}")
    print(f"  neg is dataset action ratio: {(idx_neg == 0).float().mean().item():.6f}")
    print(f"  idx_pos bincount: {torch.bincount(idx_pos, minlength=num_candidates)}")
    print(f"  idx_neg bincount: {torch.bincount(idx_neg, minlength=num_candidates)}")

    print("\nAction stats:")
    for k in ["a_ctx", "a_pos", "a_neg"]:
        v = pairs[k].float()
        print(
            f"  {k}: mean={v.mean().item():.6f}, std={v.std().item():.6f}, "
            f"min={v.min().item():.6f}, max={v.max().item():.6f}"
        )


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg: DictConfig):
    stage_start_time = time.perf_counter()
    dpo_cfg = get_dpo_cfg(cfg)

    seed = int(dpo_cfg.seed)
    torch.manual_seed(seed)

    device = str(dpo_cfg.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ctx_len = int(dpo_cfg.ctx_len)
    plan_horizon = int(dpo_cfg.plan_horizon)
    num_candidates = int(dpo_cfg.num_candidates)

    if ctx_len < 1:
        raise ValueError(f"ctx_len must be >= 1, got {ctx_len}")
    if plan_horizon < ctx_len:
        raise ValueError(
            f"plan_horizon={plan_horizon} must be >= ctx_len={ctx_len}, "
            f"because JEPA.get_cost splits action sequence by context length."
        )

    print("==== Eval-style DPO pair generation config ====")
    print(OmegaConf.to_yaml(dpo_cfg))
    print(f"ctx_len={ctx_len}, plan_horizon={plan_horizon}, device={device}")

    dataset, loader = build_evalstyle_dataset_and_loader(cfg, dpo_cfg)
    model = load_frozen_lewm(dpo_cfg, device)

    storage = {
        "z_ctx": [],
        "z_goal": [],
        "a_ctx": [],
        "a_pos": [],
        "a_neg": [],
        "cost_pos": [],
        "cost_neg": [],
        "idx_pos": [],
        "idx_neg": [],
    }

    neg_generator = torch.Generator(device=device)
    neg_generator.manual_seed(seed + 999)

    num_saved = 0
    pbar = tqdm(total=int(dpo_cfg.num_pairs), desc="Generating eval-style DPO pairs")

    for batch in loader:
        if num_saved >= int(dpo_cfg.num_pairs):
            break

        if "pixels" not in batch or "action" not in batch:
            raise KeyError(f"Batch must contain 'pixels' and 'action'. Got: {list(batch.keys())}")

        batch = move_batch_to_device(batch, device)
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        pixels = batch["pixels"]   # (B, T, C, H, W)
        actions = batch["action"]  # (B, T, A)

        if pixels.ndim != 5:
            raise ValueError(f"Expected pixels shape (B,T,C,H,W), got {tuple(pixels.shape)}")
        if actions.ndim != 3:
            raise ValueError(f"Expected action shape (B,T,A), got {tuple(actions.shape)}")

        B, T = pixels.shape[:2]

        required_T = plan_horizon + 1
        if T < required_T:
            raise ValueError(
                f"Expected dataset sequence length T >= {required_T}, got T={T}. "
                f"This script should override cfg.data.dataset.num_steps={required_T}."
            )

        pixels_ctx = pixels[:, :ctx_len]
        goal_pixels = pixels[:, plan_horizon : plan_horizon + 1]

        # Policy context action. Eval-style ctx_len=1 by default.
        a_ctx = actions[:, :ctx_len]

        # Full candidate plan length = plan_horizon.
        candidates, dataset_plan = build_candidate_plans(
            actions=actions,
            plan_horizon=plan_horizon,
            num_candidates=num_candidates,
            noise_std=float(dpo_cfg.noise_std),
            include_dataset_action=bool(dpo_cfg.include_dataset_action),
        )

        # Score via the exact eval cost path.
        cost = cost_with_model_get_cost(
            model=model,
            pixels_ctx=pixels_ctx,
            goal_pixels=goal_pixels,
            action_candidates=candidates,
        )

        idx_pos = select_positive_indices(
            cost=cost,
            pos_strategy=str(dpo_cfg.pos_strategy),
            num_candidates=num_candidates,
            include_dataset_action=bool(dpo_cfg.include_dataset_action),
        )

        idx_neg = select_negative_indices(
            cost=cost,
            idx_pos=idx_pos,
            neg_strategy=str(dpo_cfg.neg_strategy),
            generator=neg_generator,
        )

        batch_idx = torch.arange(B, device=device)

        a_pos = candidates[batch_idx, idx_pos]
        a_neg = candidates[batch_idx, idx_neg]

        cost_pos = cost[batch_idx, idx_pos]
        cost_neg = cost[batch_idx, idx_neg]

        # Store latent inputs for train_dpo.py.
        z_ctx = encode_pixels(model, pixels_ctx)
        z_goal = encode_pixels(model, goal_pixels)[:, -1]

        remaining = int(dpo_cfg.num_pairs) - num_saved
        take = min(B, remaining)

        append_batch(
            storage=storage,
            z_ctx=z_ctx,
            z_goal=z_goal,
            a_ctx=a_ctx,
            a_pos=a_pos,
            a_neg=a_neg,
            cost_pos=cost_pos,
            cost_neg=cost_neg,
            idx_pos=idx_pos,
            idx_neg=idx_neg,
            take=take,
            save_fp16=bool(dpo_cfg.save_fp16),
        )

        num_saved += take
        pbar.update(take)

    pbar.close()

    if num_saved == 0:
        raise RuntimeError("No eval-style DPO pairs were generated.")

    model_path = Path(
        swm.data.utils.get_cache_dir(),
        str(dpo_cfg.model_dir),
        str(dpo_cfg.model_file),
    )
    pair_generation_time_sec = time.perf_counter() - stage_start_time
    
    meta = {
        "stage": "make_dpo_pairs",
        "num_pairs": num_saved,
        "ctx_len": ctx_len,
        "plan_horizon": plan_horizon,
        "num_candidates": num_candidates,
        "n_noisy": num_candidates // 2,
        "noise_std": float(dpo_cfg.noise_std),
        "include_dataset_action": bool(dpo_cfg.include_dataset_action),
        "pos_strategy": str(dpo_cfg.pos_strategy),
        "neg_strategy": str(dpo_cfg.neg_strategy),
        "model_path": str(model_path),
        "pair_generation_time_sec": float(pair_generation_time_sec),
        "pair_generation_time_min": float(pair_generation_time_sec / 60.0),
        "time_per_pair_sec": float(pair_generation_time_sec / max(num_saved, 1)),
        "data_config": OmegaConf.to_container(cfg.data, resolve=True),
        "wm_config": OmegaConf.to_container(cfg.wm, resolve=True),
        "note": "Eval-style pairs generated using JEPA.get_cost with action plan length equal to plan_horizon.",
    }

    output_path = Path(swm.data.utils.get_cache_dir(), str(dpo_cfg.output))
    finalize_and_save(storage, meta, output_path)


if __name__ == "__main__":
    run()