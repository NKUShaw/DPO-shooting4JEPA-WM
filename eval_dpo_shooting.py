import os

os.environ["MUJOCO_GL"] = "egl"

import json
import time
from pathlib import Path
from collections import deque
from typing import Any

import gymnasium as gym
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from gymnasium.spaces import Box
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from train_dpo_plan import GaussianPlanPolicyNoAct

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()

    if isinstance(obj, Path):
        return str(obj)

    return obj
    
def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def freeze_model(model: torch.nn.Module):
    model.eval()
    model.requires_grad_(False)
    return model


def load_dpo_policy(policy_path: Path, device: str):
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


class DPOShootingSolver:
    """
    DPO shooting solver.

    It replaces CEM sampling with samples from a trained DPO planner:
        z_ctx, z_goal -> sample K action plans -> LeWM cost rerank -> return best plan.

    Compatible with swm.policy.WorldModelPolicy because it implements:
        configure(...)
        __call__(info_dict, init_action=None)
        solve(...)
    """

    def __init__(
        self,
        model,
        dpo_policy_path: str,
        lewm_object_path: str,
        batch_size: int = 1,
        num_samples: int = 128,
        random_frac: float = 0.0,
        include_mean: bool = True,
        device: str | torch.device = "cuda",
        seed: int = 42,
    ):
        self.model = model
        self.batch_size = batch_size
        self.num_samples = int(num_samples)
        self.random_frac = float(random_frac)
        self.include_mean = bool(include_mean)
        self.device = torch.device(device)
        self.seed = int(seed)

        self.torch_gen = torch.Generator(device=self.device).manual_seed(self.seed)

        self.dpo_policy_path = Path(dpo_policy_path)
        self.lewm_object_path = Path(lewm_object_path)

        if not self.dpo_policy_path.exists():
            raise FileNotFoundError(f"Cannot find DPO policy: {self.dpo_policy_path}")
        if not self.lewm_object_path.exists():
            raise FileNotFoundError(f"Cannot find LeWM object: {self.lewm_object_path}")

        print(f"[DPOShootingSolver] Loading DPO policy from: {self.dpo_policy_path}")
        self.dpo_policy, self.dpo_payload = load_dpo_policy(self.dpo_policy_path, str(self.device))

        print(f"[DPOShootingSolver] Loading LeWM object from: {self.lewm_object_path}")
        self.lewm_model = torch.load(
            self.lewm_object_path,
            map_location="cpu",
            weights_only=False,
        )
        self.lewm_model = freeze_model(self.lewm_model).to(self.device)

        self._configured = False

        self.solve_times = []
        self.total_solver_time_sec = 0.0
        self.num_solver_calls = 0
        
    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config

        self._env_action_dim = int(np.prod(action_space.shape[1:]))
        self._action_dim = self._env_action_dim * self._config.action_block
        self._configured = True

        if not isinstance(action_space, Box):
            print(f"[Warning] Action space is not Box: {type(action_space)}")

        policy_horizon = int(self.dpo_policy.horizon)
        policy_action_dim = int(self.dpo_policy.action_dim)

        if policy_horizon != self.horizon:
            raise ValueError(
                f"DPO policy horizon={policy_horizon}, but PlanConfig horizon={self.horizon}"
            )

        if policy_action_dim != self.action_dim:
            raise ValueError(
                f"DPO policy action_dim={policy_action_dim}, but solver action_dim={self.action_dim}. "
                f"Expected env_action_dim({self._env_action_dim}) * action_block({self._config.action_block})."
            )

        print("[DPOShootingSolver] configured")
        print(f"  n_envs       : {self.n_envs}")
        print(f"  horizon      : {self.horizon}")
        print(f"  action_dim   : {self.action_dim}")
        print(f"  num_samples  : {self.num_samples}")
        print(f"  random_frac  : {self.random_frac}")
        print(f"  include_mean : {self.include_mean}")

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    @torch.inference_mode()
    def _encode_context_goal(self, info_batch: dict):
        """
        info_batch contains prepared tensors from WorldModelPolicy._prepare_info.

        Expected:
            pixels: (B, ctx_len, C, H, W)
            goal:   (B, C, H, W) or (B, 1, C, H, W)
        """
        pixels = info_batch["pixels"]
        goal = info_batch["goal"]

        if not torch.is_tensor(pixels):
            pixels = torch.as_tensor(pixels)
        if not torch.is_tensor(goal):
            goal = torch.as_tensor(goal)

        pixels = pixels.to(self.device, non_blocking=True).float()
        goal = goal.to(self.device, non_blocking=True).float()

        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)

        if goal.ndim == 4:
            goal = goal.unsqueeze(1)

        z_ctx = self.lewm_model.encode({"pixels": pixels})["emb"]
        z_goal = self.lewm_model.encode({"pixels": goal})["emb"][:, -1]

        return z_ctx, z_goal

    def _expand_info_for_cost(self, info_batch: dict, num_samples: int):
        """
        Mimic CEMSolver's expansion:
            v: (B, ...) -> (B, S, ...)
        """
        expanded = {}
        current_bs = None

        for k, v in info_batch.items():
            if torch.is_tensor(v):
                current_bs = v.shape[0]
                v = v.to(self.device, non_blocking=True)
                expanded[k] = v.unsqueeze(1).expand(current_bs, num_samples, *v.shape[1:])
            elif isinstance(v, np.ndarray):
                current_bs = v.shape[0]
                expanded[k] = np.repeat(v[:, None, ...], num_samples, axis=1)
            else:
                expanded[k] = v

        return expanded

    @torch.inference_mode()
    def _sample_candidates(self, z_ctx: torch.Tensor, z_goal: torch.Tensor):
        """
        Return:
            candidates: (B, S, H, A)
        """
        B = z_ctx.shape[0]
        S = self.num_samples

        if S <= 0:
            raise ValueError(f"num_samples must be > 0, got {S}")

        n_random = int(round(S * self.random_frac))
        n_random = max(0, min(S, n_random))

        n_policy = S - n_random

        chunks = []

        if n_policy > 0:
            policy_actions, _ = self.dpo_policy.sample(
                z_ctx=z_ctx,
                z_goal=z_goal,
                num_samples=n_policy,
            )
            chunks.append(policy_actions)

        if n_random > 0:
            random_actions = torch.randn(
                B,
                n_random,
                self.horizon,
                self.action_dim,
                generator=self.torch_gen,
                device=self.device,
            )
            chunks.append(random_actions)

        candidates = torch.cat(chunks, dim=1)

        if self.include_mean:
            mean_action, _ = self.dpo_policy(z_ctx, z_goal)
            candidates[:, 0] = mean_action

        return candidates

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.perf_counter()

        total_envs = self.n_envs
        all_actions = []
        all_costs = []

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx

            info_batch = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    info_batch[k] = v[start_idx:end_idx]
                elif isinstance(v, np.ndarray):
                    info_batch[k] = v[start_idx:end_idx]
                else:
                    info_batch[k] = v

            z_ctx, z_goal = self._encode_context_goal(info_batch)

            candidates = self._sample_candidates(z_ctx, z_goal)
            num_samples = candidates.shape[1]

            expanded_info = self._expand_info_for_cost(info_batch, num_samples=num_samples)

            costs = self.model.get_cost(expanded_info, candidates)

            assert isinstance(costs, torch.Tensor), f"Expected tensor costs, got {type(costs)}"
            assert costs.shape == (current_bs, num_samples), (
                f"Expected costs shape {(current_bs, num_samples)}, got {tuple(costs.shape)}"
            )

            best_idx = costs.argmin(dim=1)
            batch_idx = torch.arange(current_bs, device=self.device)
            best_actions = candidates[batch_idx, best_idx]

            best_costs = costs[batch_idx, best_idx]

            all_actions.append(best_actions.detach().cpu())
            all_costs.extend(best_costs.detach().cpu().tolist())

        actions = torch.cat(all_actions, dim=0)

        outputs = {
            "actions": actions,
            "costs": all_costs,
            "mean": [actions],
            "var": [],
        }

        solve_time = time.perf_counter() - start_time

        self.solve_times.append(float(solve_time))
        self.total_solver_time_sec += float(solve_time)
        self.num_solver_calls += 1

        print(
            f"DPO shooting solve time: {solve_time:.4f} seconds | "
            f"mean best cost: {float(np.mean(all_costs)):.4f}"
        )

        return outputs

    def get_timing_summary(self) -> dict:
        solve_times = np.array(self.solve_times, dtype=np.float64)

        if len(solve_times) == 0:
            return {
                "num_solver_calls": 0,
                "total_solver_time_sec": 0.0,
                "avg_solver_time_sec": 0.0,
                "std_solver_time_sec": 0.0,
                "min_solver_time_sec": 0.0,
                "max_solver_time_sec": 0.0,
                "median_solver_time_sec": 0.0,
            }

        return {
            "num_solver_calls": int(self.num_solver_calls),
            "total_solver_time_sec": float(self.total_solver_time_sec),
            "avg_solver_time_sec": float(solve_times.mean()),
            "std_solver_time_sec": float(solve_times.std()),
            "min_solver_time_sec": float(solve_times.min()),
            "max_solver_time_sec": float(solve_times.max()),
            "median_solver_time_sec": float(np.median(solve_times)),
        }
        

def get_dpo_cfg(cfg: DictConfig):
    default = OmegaConf.create(
        {
            "model_dir": "cube",
            "model_file": "lewm_object.ckpt",

            "policy_dir": "dpo_plan_eval_h5_dataset_5k_bc005",
            "policy_file": "dpo_policy.pt",

            "num_samples": 128,
            "random_frac": 0.0,
            "include_mean": True,
            "batch_size": 1,
            "device": "cuda",

            "results_dir": "dpo_shooting_cube",
            "seed": 42,
        }
    )

    if "dpo_shooting" in cfg:
        return OmegaConf.merge(default, cfg.dpo_shooting)

    return default


@hydra.main(version_base=None, config_path="./config/eval", config_name="cube")
def run(cfg: DictConfig):
    """
    Eval DPO shooting planner on OGBench-Cube.

    Example:
        export STABLEWM_HOME=./WorldModels/le-wm/datasets

        python eval_dpo_shooting.py \
          +dpo_shooting.model_dir=cube \
          +dpo_shooting.policy_dir=dpo_plan_eval_h5_dataset_5k_bc005 \
          +dpo_shooting.num_samples=128 \
          +dpo_shooting.random_frac=0.0 \
          eval.num_eval=5
    """
    dpo_cfg = get_dpo_cfg(cfg)

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    cache_dir = Path(swm.data.utils.get_cache_dir())

    model_dir = cache_dir / str(dpo_cfg.model_dir)
    model_object_path = model_dir / str(dpo_cfg.model_file)

    policy_path = cache_dir / str(dpo_cfg.policy_dir) / str(dpo_cfg.policy_file)

    print("==== DPO Shooting Eval Config ====")
    print(OmegaConf.to_yaml(dpo_cfg))
    print("cache_dir        :", cache_dir)
    print("model_object_path:", model_object_path)
    print("policy_path      :", policy_path)

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create transforms
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue

        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # load LeWM cost model same as official eval
    cost_model = swm.policy.AutoCostModel(str(dpo_cfg.model_dir))
    cost_model = cost_model.to(str(dpo_cfg.device))
    cost_model = cost_model.eval()
    cost_model.requires_grad_(False)
    cost_model.interpolate_pos_encoding = True

    config = swm.PlanConfig(**cfg.plan_config)

    solver = DPOShootingSolver(
        model=cost_model,
        dpo_policy_path=str(policy_path),
        lewm_object_path=str(model_object_path),
        batch_size=int(dpo_cfg.batch_size),
        num_samples=int(dpo_cfg.num_samples),
        random_frac=float(dpo_cfg.random_frac),
        include_mean=bool(dpo_cfg.include_mean),
        device=str(dpo_cfg.device),
        seed=int(dpo_cfg.seed),
    )

    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )

    results_path = cache_dir / str(dpo_cfg.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # sample episodes and starting indices, same as eval.py
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1,
        size=cfg.eval.num_eval,
        replace=False,
    )

    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print("random_episode_indices:", random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    eval_start_time = time.perf_counter()

    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )

    eval_total_time_sec = time.perf_counter() - eval_start_time

    try:
        if hasattr(world, "close"):
            world.close()
        elif hasattr(world, "env") and hasattr(world.env, "close"):
            world.env.close()
    except Exception as e:
        print(f"[Warning] Failed to close world cleanly: {e}")
        
    solver_timing = solver.get_timing_summary()

    print(metrics)
    print("\n==== DPO SHOOTING TIMING ====")
    print(f"total_eval_time_sec   : {eval_total_time_sec:.4f}")
    print(f"num_solver_calls      : {solver_timing['num_solver_calls']}")
    print(f"total_solver_time_sec : {solver_timing['total_solver_time_sec']:.4f}")
    print(f"avg_solver_time_sec   : {solver_timing['avg_solver_time_sec']:.6f}")
    print(f"median_solver_time_sec: {solver_timing['median_solver_time_sec']:.6f}")

    output_path = results_path / cfg.output.filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timing = {
        "stage": "eval_dpo_shooting",
        "results_dir": str(results_path),
        "model_object_path": str(model_object_path),
        "policy_path": str(policy_path),
        "num_eval": int(cfg.eval.num_eval),
        "eval_budget": int(cfg.eval.eval_budget),
        "goal_offset_steps": int(cfg.eval.goal_offset_steps),
        "num_samples": int(dpo_cfg.num_samples),
        "random_frac": float(dpo_cfg.random_frac),
        "include_mean": bool(dpo_cfg.include_mean),
        "batch_size": int(dpo_cfg.batch_size),
        "total_eval_time_sec": float(eval_total_time_sec),
        "total_eval_time_min": float(eval_total_time_sec / 60.0),
        **solver_timing,
        "metrics": make_json_serializable(metrics),
    }
    
    timing_path = results_path / "timing.json"
    with timing_path.open("w") as f:
        json.dump(make_json_serializable(timing), f, indent=2)
        
    with output_path.open("a") as f:
        f.write("\n")
        f.write("==== DPO SHOOTING CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== DPO SHOOTING EXTRA CONFIG ====\n")
        f.write(OmegaConf.to_yaml(dpo_cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write("\n")
        f.write("==== TIMING ====\n")
        f.write(f"total_eval_time_sec: {eval_total_time_sec}\n")
        f.write(f"total_eval_time_min: {eval_total_time_sec / 60.0}\n")
        f.write(f"num_solver_calls: {solver_timing['num_solver_calls']}\n")
        f.write(f"total_solver_time_sec: {solver_timing['total_solver_time_sec']}\n")
        f.write(f"avg_solver_time_sec: {solver_timing['avg_solver_time_sec']}\n")
        f.write(f"median_solver_time_sec: {solver_timing['median_solver_time_sec']}\n")
        f.write(f"min_solver_time_sec: {solver_timing['min_solver_time_sec']}\n")
        f.write(f"max_solver_time_sec: {solver_timing['max_solver_time_sec']}\n")
        f.write(f"std_solver_time_sec: {solver_timing['std_solver_time_sec']}\n")
        f.write(f"timing_json: {timing_path}\n")

    print(f"Saved results to: {output_path}")
    print(f"Saved timing to: {timing_path}")


if __name__ == "__main__":
    run()