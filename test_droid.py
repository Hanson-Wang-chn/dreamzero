#!/usr/bin/env python3
"""Compare video-only flow-matching loss on DROID with/without GT actions.

This script evaluates a few fixed DROID samples with the DreamZero-DROID
checkpoint and compares two video flow-matching settings:

1. Standard mode: action tokens follow the original flow-matching path and are
   noised before being passed into the video model.
2. GT-action mode: the same video noise is used, but clean normalized GT
   actions are fed into the model with action timestep 0.

Only the video prediction loss is reported. The action loss is intentionally
excluded because GT-action conditioning would make that comparison unfair.
"""

import torch._dynamo

torch._dynamo.config.disable = True

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
from hydra.utils import instantiate
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import collate
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


# ---------------------------------------------------------------------------
# Edit these fixed samples if you want to test different DROID clips.
# Each entry is (episode_index, base_index, note).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleSpec:
    episode_index: int
    base_index: int
    note: str = ""


DEFAULT_SAMPLE_SPECS = [
    SampleSpec(episode_index=0, base_index=24, note="early-episode-0"),
    SampleSpec(episode_index=1, base_index=24, note="early-episode-1"),
    SampleSpec(episode_index=2, base_index=24, note="early-episode-2"),
    SampleSpec(episode_index=15, base_index=24, note="brush-pan"),
    SampleSpec(episode_index=42, base_index=24, note="mid-sample"),
]


DROID_VIDEO_KEYS = {
    "video.exterior_image_1_left": "observation.images.exterior_image_1_left",
    "video.exterior_image_2_left": "observation.images.exterior_image_2_left",
    "video.wrist_image_left": "observation.images.wrist_image_left",
}

LANGUAGE_KEYS = [
    "annotation.language.language_instruction",
    "annotation.language.language_instruction_2",
    "annotation.language.language_instruction_3",
]

STATE_SLICES = {
    "state.joint_position": slice(7, 14),
    "state.gripper_position": slice(6, 7),
}

ACTION_SLICES = {
    "action.joint_position": slice(14, 21),
    "action.gripper_position": slice(12, 13),
}

# The training-time teacher-forcing branch used below expects a single
# video/action block. For DreamZero-DROID with action_horizon=24 and
# num_frame_per_block=2, this corresponds to 9 raw frames:
#   1 conditioning frame + 8 future raw frames -> 3 latent frames after VAE.
VIDEO_DELTAS = np.arange(9, dtype=np.int64)
ACTION_DELTAS = np.arange(24, dtype=np.int64)


# ---------------------------------------------------------------------------
# Dataset reader
# ---------------------------------------------------------------------------


class DROIDDataset:
    """Reads a small number of DROID samples from LeRobot-format parquet + MP4."""

    def __init__(self, dataset_path: str):
        self.root = Path(dataset_path)
        if not self.root.exists():
            raise FileNotFoundError(f"DROID dataset not found: {self.root}")

        self._episode_cache: dict[int, pd.DataFrame] = {}
        self._task_by_index = self._load_tasks()

    def _load_tasks(self) -> dict[int, str]:
        tasks_path = self.root / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(f"tasks.jsonl not found: {tasks_path}")

        task_by_index: dict[int, str] = {}
        with open(tasks_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                task_by_index[int(item["task_index"])] = str(item["task"])
        return task_by_index

    def _get_episode_df(self, episode_index: int) -> pd.DataFrame:
        if episode_index not in self._episode_cache:
            chunk_index = episode_index // 1000
            parquet_path = (
                self.root
                / "data"
                / f"chunk-{chunk_index:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
            if not parquet_path.exists():
                raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
            self._episode_cache[episode_index] = pd.read_parquet(parquet_path)
        return self._episode_cache[episode_index]

    def _clip_indices(self, indices: np.ndarray, length: int) -> np.ndarray:
        return np.clip(indices, 0, max(length - 1, 0)).astype(np.int64)

    def _read_video_frames(
        self,
        episode_index: int,
        original_video_key: str,
        frame_indices: np.ndarray,
    ) -> np.ndarray:
        chunk_index = episode_index // 1000
        video_path = (
            self.root
            / "videos"
            / f"chunk-{chunk_index:03d}"
            / original_video_key
            / f"episode_{episode_index:06d}.mp4"
        )
        if not video_path.exists():
            raise FileNotFoundError(f"Episode video not found: {video_path}")

        cap = cv2.VideoCapture(video_path.as_posix())
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frames: list[np.ndarray] = []
        current_index = -1
        for frame_index in frame_indices:
            frame_index = int(frame_index)
            if frame_index != current_index + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_index = frame_index

        cap.release()
        return np.stack(frames, axis=0)

    def _get_prompt(self, episode_df: pd.DataFrame, row_index: int) -> str:
        row = episode_df.iloc[row_index]
        for key in LANGUAGE_KEYS:
            if key not in episode_df.columns:
                continue
            task_index = int(row[key])
            task = self._task_by_index.get(task_index, "")
            if task and task != "not provided":
                return task
        return "Perform the default behavior."

    def build_raw_sample(self, spec: SampleSpec, prompt_override: str | None = None) -> dict[str, Any]:
        episode_df = self._get_episode_df(spec.episode_index)
        episode_len = len(episode_df)
        if episode_len <= 0:
            raise ValueError(f"Episode {spec.episode_index} is empty")

        state_index = min(max(spec.base_index, 0), episode_len - 1)
        video_indices = self._clip_indices(spec.base_index + VIDEO_DELTAS, episode_len)
        action_indices = self._clip_indices(spec.base_index + ACTION_DELTAS, episode_len)

        state_row = np.asarray(episode_df.iloc[state_index]["observation.state"], dtype=np.float64)
        action_rows = np.stack(
            [np.asarray(x, dtype=np.float64) for x in episode_df.iloc[action_indices]["action"].tolist()],
            axis=0,
        )

        raw_sample: dict[str, Any] = {}
        for key, original_video_key in DROID_VIDEO_KEYS.items():
            raw_sample[key] = self._read_video_frames(
                spec.episode_index,
                original_video_key,
                video_indices,
            )

        for key, state_slice in STATE_SLICES.items():
            raw_sample[key] = state_row[state_slice].reshape(1, -1).astype(np.float64)

        for key, action_slice in ACTION_SLICES.items():
            raw_sample[key] = action_rows[:, action_slice].astype(np.float64)

        raw_sample["annotation.language.language_instruction"] = (
            prompt_override if prompt_override is not None else self._get_prompt(episode_df, state_index)
        )
        return raw_sample


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def init_runtime(device_arg: str, master_port: int) -> tuple[int, int, int, torch.device]:
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(device_arg)
        return rank, world_size, local_rank, device

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU evaluation requires CUDA.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return dist.get_rank(), dist.get_world_size(), local_rank, torch.device(f"cuda:{local_rank}")

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
    device = torch.device(device_arg)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return 0, 1, 0, device


def cleanup_runtime():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def shard_samples(sample_specs: list[SampleSpec], rank: int, world_size: int) -> list[tuple[int, SampleSpec]]:
    return [(i, spec) for i, spec in enumerate(sample_specs) if i % world_size == rank]


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Transform + loss helpers
# ---------------------------------------------------------------------------


def build_training_transform(policy: GrootSimPolicy):
    transform_cfg = policy.train_cfg.transforms[EmbodimentTag.OXE_DROID.value]
    train_transform = instantiate(transform_cfg)
    metadata = policy.eval_transform.transforms[0].dataset_metadata
    train_transform.set_metadata(metadata)
    train_transform.train()
    return train_transform


def prepare_action_inputs(
    policy: GrootSimPolicy,
    train_transform,
    raw_sample: dict[str, Any],
):
    transformed = train_transform(raw_sample)
    dream_transform = train_transform.transforms[-1]
    if not hasattr(dream_transform, "tokenizer"):
        raise TypeError(
            "Expected the last transform to be DreamTransform with tokenizer support, "
            f"got {type(dream_transform).__name__}"
        )
    batched = collate(
        [transformed],
        tokenizer=dream_transform.tokenizer,
        num_views=dream_transform.num_views,
        embodiment_tag_mapping=dream_transform.embodiment_tag_mapping,
    )
    backbone_inputs, action_inputs = policy.trained_model.prepare_input(batched)
    backbone_outputs = policy.trained_model.backbone(backbone_inputs)
    return backbone_outputs, action_inputs


def _normalize_videos(action_head, videos: torch.Tensor) -> torch.Tensor:
    videos = videos.float() / 255.0
    videos = videos.to(dtype=action_head.dtype)
    bsz, channels, num_frames, height, width = videos.shape
    videos = videos.permute(0, 2, 1, 3, 4)
    videos = videos.reshape(bsz * num_frames, channels, height, width)
    videos = action_head.normalize_video(videos)
    videos = videos.reshape(bsz, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
    return videos.to(dtype=action_head.dtype)


def _sample_video_and_action_timesteps(action_head, noise: torch.Tensor, actions: torch.Tensor):
    if action_head.config.decouple_video_action_noise:
        video_noise_ratio = action_head.video_beta_dist.sample([noise.shape[0], noise.shape[1]])
        timestep_id = ((1.0 - video_noise_ratio) * action_head.scheduler.num_train_timesteps).long()
        timestep_id = torch.clamp(timestep_id, 0, action_head.scheduler.num_train_timesteps - 1)
    elif action_head.config.use_high_noise_emphasis:
        noise_ratio = action_head.high_noise_beta_dist.sample([noise.shape[0], noise.shape[1]])
        timestep_id = ((1.0 - noise_ratio) * action_head.scheduler.num_train_timesteps).long()
        timestep_id = torch.clamp(timestep_id, 0, action_head.scheduler.num_train_timesteps - 1)
    else:
        timestep_id = torch.randint(
            0,
            action_head.scheduler.num_train_timesteps,
            (noise.shape[0], noise.shape[1]),
        )

    timestep_id_block = timestep_id[:, 1:].reshape(
        timestep_id.shape[0],
        -1,
        action_head.num_frame_per_block,
    )
    timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]

    if action_head.config.decouple_video_action_noise:
        timestep_action_id = torch.randint(
            0,
            action_head.scheduler.num_train_timesteps,
            (actions.shape[0], actions.shape[1]),
        )
    else:
        timestep_action_id = timestep_id_block.repeat(
            1,
            1,
            actions.shape[1] // (noise.shape[1] - 1),
        )
        timestep_action_id = timestep_action_id.reshape(timestep_action_id.shape[0], -1)

    timestep_id_block = timestep_id_block.reshape(timestep_id_block.shape[0], -1)
    timestep_id = torch.cat([timestep_id[:, :1], timestep_id_block], dim=1)
    return timestep_id, timestep_action_id


def _compute_video_loss_terms(
    action_head,
    video_noise_pred: torch.Tensor,
    training_target: torch.Tensor,
    timestep: torch.Tensor,
    noise_shape: torch.Size,
) -> tuple[float, float]:
    raw_loss_per_token = torch.nn.functional.mse_loss(
        video_noise_pred.float(),
        training_target.float(),
        reduction="none",
    ).mean(dim=(1, 3, 4))
    raw_loss = float(raw_loss_per_token.mean().item())

    weights = action_head.scheduler.training_weight(timestep.flatten(0, 1))
    weights = weights.unflatten(0, (noise_shape[0], noise_shape[1])).to(video_noise_pred.device)
    weighted_loss = float((raw_loss_per_token * weights).mean().item())
    return raw_loss, weighted_loss


def compute_video_flow_metrics(
    policy: GrootSimPolicy,
    backbone_outputs,
    action_inputs,
) -> dict[str, float]:
    del backbone_outputs  # IdentityBackbone output is not used by the action head.

    action_head = policy.trained_model.action_head
    action_head.set_frozen_modules_to_eval_mode()

    data = action_inputs
    actions = data["action"]
    state_features = data["state"].to(dtype=torch.bfloat16)
    embodiment_id = data["embodiment_id"]
    videos = data["images"]

    if videos.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 images after transform, got {videos.dtype}")

    videos = videos.permute(0, 4, 1, 2, 3)
    videos = _normalize_videos(action_head, videos)

    with torch.inference_mode():
        prompt_embs = action_head.encode_prompt(data["text"], data["text_attention_mask"]).to(action_head._device)

        latents = action_head.encode_video(
            videos,
            tiled=action_head.tiled,
            tile_size=(action_head.tile_size_height, action_head.tile_size_width),
            tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
        )

        _, _, num_frames, height, width = videos.shape
        image = videos[:, :, :1].transpose(1, 2)
        clip_feas, ys, _ = action_head.encode_image(image, num_frames, height, width)

        latents = latents.to(action_head._device)
        clip_feas = clip_feas.to(action_head._device)
        ys = ys.to(action_head._device)

        noise = torch.randn_like(latents)
        noise = noise.transpose(1, 2)
        latents = latents.transpose(1, 2)

        if noise.shape[1] not in (3, 4):
            raise ValueError(
                "Unexpected latent video length for teacher-forcing loss. "
                "This script expects a single video block so that the internal "
                "action/state token layout matches action_horizon=24. "
                f"Got latent_frames={noise.shape[1]} from videos.shape={videos.shape}."
            )
        assert actions.shape[1] % (noise.shape[1] - 1) == 0, (
            f"Action horizon / latent steps mismatch: {actions.shape=} vs {noise.shape=}"
        )
        timestep_id, timestep_action_id = _sample_video_and_action_timesteps(action_head, noise, actions)
        timestep = action_head.scheduler.timesteps[timestep_id].to(action_head._device)

        noisy_latents = action_head.scheduler.add_noise(
            latents.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (noise.shape[0], noise.shape[1]))
        training_target = action_head.scheduler.training_target(latents, noise, timestep).transpose(1, 2)

        timestep_action = action_head.scheduler.timesteps[timestep_action_id].to(action_head._device)
        noise_action = torch.randn_like(actions)
        noisy_actions = action_head.scheduler.add_noise(
            actions.flatten(0, 1),
            noise_action.flatten(0, 1),
            timestep_action.flatten(0, 1),
        ).unflatten(0, (actions.shape[0], actions.shape[1]))
        gt_timestep_action = torch.zeros_like(timestep_action)

        frame_seqlen = int(noise.shape[-2] * noise.shape[-1] / 4)
        seq_len = noise.shape[1] * frame_seqlen

        autocast_device = torch.device(action_head._device).type
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=autocast_device):
            video_noise_pred_standard, _ = action_head.model(
                noisy_latents.transpose(1, 2),
                timestep=timestep,
                clip_feature=clip_feas,
                y=ys,
                context=prompt_embs,
                seq_len=seq_len,
                state=state_features,
                embodiment_id=embodiment_id,
                action=noisy_actions,
                timestep_action=timestep_action,
                clean_x=latents.transpose(1, 2),
            )
            video_noise_pred_gt, _ = action_head.model(
                noisy_latents.transpose(1, 2),
                timestep=timestep,
                clip_feature=clip_feas,
                y=ys,
                context=prompt_embs,
                seq_len=seq_len,
                state=state_features,
                embodiment_id=embodiment_id,
                action=actions,
                timestep_action=gt_timestep_action,
                clean_x=latents.transpose(1, 2),
            )

        standard_raw, standard_weighted = _compute_video_loss_terms(
            action_head,
            video_noise_pred_standard,
            training_target,
            timestep,
            noise.shape,
        )
        gt_raw, gt_weighted = _compute_video_loss_terms(
            action_head,
            video_noise_pred_gt,
            training_target,
            timestep,
            noise.shape,
        )

    return {
        "video_l2_raw_standard": standard_raw,
        "video_l2_raw_gt_action": gt_raw,
        "video_l2_raw_gt_minus_standard": gt_raw - standard_raw,
        "video_l2_weighted_standard": standard_weighted,
        "video_l2_weighted_gt_action": gt_weighted,
        "video_l2_weighted_gt_minus_standard": gt_weighted - standard_weighted,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"num_samples": 0}

    keys = [
        "video_l2_raw_standard",
        "video_l2_raw_gt_action",
        "video_l2_raw_gt_minus_standard",
        "video_l2_weighted_standard",
        "video_l2_weighted_gt_action",
        "video_l2_weighted_gt_minus_standard",
        "elapsed_seconds",
    ]
    summary = {"num_samples": len(results)}
    for key in keys:
        values = np.array([r[key] for r in results], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
        summary[f"{key}_min"] = float(values.min())
        summary[f"{key}_max"] = float(values.max())
    return summary


def print_results(results: list[dict[str, Any]], summary: dict[str, Any]):
    print("\n" + "=" * 120)
    print("Per-sample video flow-matching loss comparison")
    print("=" * 120)
    print(
        f"{'sample':>6}  {'episode':>8}  {'base':>6}  "
        f"{'raw/no-gt':>12}  {'raw/gt':>12}  {'raw diff':>12}  "
        f"{'weighted/no-gt':>16}  {'weighted/gt':>14}  {'weighted diff':>14}"
    )
    print("-" * 120)
    for item in results:
        print(
            f"{item['sample_id']:>6}  {item['episode_index']:>8}  {item['base_index']:>6}  "
            f"{item['video_l2_raw_standard']:>12.6f}  "
            f"{item['video_l2_raw_gt_action']:>12.6f}  "
            f"{item['video_l2_raw_gt_minus_standard']:>12.6f}  "
            f"{item['video_l2_weighted_standard']:>16.6f}  "
            f"{item['video_l2_weighted_gt_action']:>14.6f}  "
            f"{item['video_l2_weighted_gt_minus_standard']:>14.6f}"
        )
    print("-" * 120)
    print(
        "Means: "
        f"raw/no-gt={summary['video_l2_raw_standard_mean']:.6f}, "
        f"raw/gt={summary['video_l2_raw_gt_action_mean']:.6f}, "
        f"raw diff={summary['video_l2_raw_gt_minus_standard_mean']:.6f}, "
        f"weighted/no-gt={summary['video_l2_weighted_standard_mean']:.6f}, "
        f"weighted/gt={summary['video_l2_weighted_gt_action_mean']:.6f}, "
        f"weighted diff={summary['video_l2_weighted_gt_minus_standard_mean']:.6f}"
    )
    print("=" * 120 + "\n")


def save_results(
    output_dir: Path,
    args,
    sample_specs: list[SampleSpec],
    results: list[dict[str, Any]],
    summary: dict[str, Any],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_path = output_dir / "per_sample.jsonl"
    with open(per_sample_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    payload = {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "device": args.device,
        "seed": args.seed,
        "sample_specs": [asdict(spec) for spec in sample_specs],
        "summary": summary,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(args):
    rank, world_size, _local_rank, device = init_runtime(args.device, args.master_port)
    if device.type != "cuda":
        raise RuntimeError("DreamZero-DROID flow-matching evaluation currently requires CUDA.")

    if rank == 0:
        print(f"Loading DreamZero-DROID checkpoint from {args.model_path}")
        print(f"Using dataset: {args.dataset_path}")
        print(f"World size: {world_size}")

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.OXE_DROID,
        model_path=args.model_path,
        device=str(device),
    )
    policy.trained_model.eval()
    policy.trained_model.requires_grad_(False)

    train_transform = build_training_transform(policy)
    dataset = DROIDDataset(args.dataset_path)
    local_results: list[dict[str, Any]] = []

    assigned_specs = shard_samples(DEFAULT_SAMPLE_SPECS, rank, world_size)
    for sample_id, spec in assigned_specs:
        seed = args.seed + sample_id
        set_global_seed(seed)

        start = time.perf_counter()
        raw_sample = dataset.build_raw_sample(spec, prompt_override=args.prompt)
        backbone_outputs, action_inputs = prepare_action_inputs(policy, train_transform, raw_sample)
        metrics = compute_video_flow_metrics(policy, backbone_outputs, action_inputs)
        elapsed = time.perf_counter() - start

        local_results.append(
            {
                "sample_id": sample_id,
                "episode_index": spec.episode_index,
                "base_index": spec.base_index,
                "note": spec.note,
                "prompt": raw_sample["annotation.language.language_instruction"],
                "elapsed_seconds": float(elapsed),
                **metrics,
            }
        )
        print(
            f"[rank {rank}] sample={sample_id} episode={spec.episode_index} base={spec.base_index} "
            f"weighted_diff={metrics['video_l2_weighted_gt_minus_standard']:.6f} "
            f"elapsed={elapsed:.2f}s"
        )

    gathered: list[list[dict[str, Any]]] = [None for _ in range(world_size)]  # type: ignore
    dist.all_gather_object(gathered, local_results)

    if rank == 0:
        results = [item for sublist in gathered for item in sublist]
        results.sort(key=lambda x: x["sample_id"])
        summary = summarize_results(results)
        print_results(results, summary)
        save_results(Path(args.output_dir), args, DEFAULT_SAMPLE_SPECS, results, summary)
        print(f"Saved results to {Path(args.output_dir).resolve()}")

    cleanup_runtime()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model_path",
        default="./checkpoints",
        help="Path to the DreamZero-DROID checkpoint directory",
    )
    parser.add_argument(
        "--dataset_path",
        default="/mnt/data/dataset/lerobot/GEAR-Dreams/DreamZero-DROID-Data",
        help="Root of the DROID LeRobot dataset",
    )
    parser.add_argument("--device", default="cuda:0", help="Single-process device to use")
    parser.add_argument(
        "--output_dir",
        default="results/test_droid",
        help="Directory used to save per-sample and summary metrics",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt override. By default the dataset instruction is used.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Base random seed")
    parser.add_argument(
        "--master_port",
        type=int,
        default=29611,
        help="Port used for single-process fallback distributed init",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
