import argparse
import time
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf

from train import TrainDP3Workspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


def find_latest_checkpoint(root: Path) -> Path:
    candidates = list(root.glob("**/checkpoints/latest.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No latest.ckpt found under: {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_random_obs(cfg, batch_size: int, n_obs_steps: int, device: torch.device) -> Dict[str, torch.Tensor]:
    obs_dict = {}
    obs_meta = cfg.shape_meta.obs
    for key, meta in obs_meta.items():
        shape = tuple(int(x) for x in meta.shape)
        tensor_shape = (batch_size, n_obs_steps, *shape)
        obs_dict[key] = torch.randn(tensor_shape, dtype=torch.float32, device=device)
    return obs_dict


def main():
    default_checkpoint = "3D-Diffusion-Policy/data/outputs/adroit_hammer-simple_dp3-0306_seed0/checkpoints/latest.ckpt"
    parser = argparse.ArgumentParser(description="使用随机输入测试策略模型的 predict_action 输出维度。")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_checkpoint,
        help=f"checkpoint 路径（.ckpt），默认: {default_checkpoint}",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="随机观测的 batch 大小。")
    parser.add_argument(
        "--obs-steps",
        type=int,
        default=None,
        help="覆盖 n_obs_steps（默认使用配置文件中的 n_obs_steps）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备，例如 cpu / cuda / cuda:0",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只运行一次（默认以 20Hz 无限循环）。",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(Path("data/outputs"))
    output_dir = ckpt_path.parent.parent
    cfg_path = output_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    workspace = TrainDP3Workspace(cfg, output_dir=str(output_dir))
    workspace.load_checkpoint(path=ckpt_path)

    policy = workspace.ema_model if cfg.training.use_ema and workspace.ema_model is not None else workspace.model
    device = torch.device(args.device)
    policy.to(device).eval()

    n_obs_steps = int(args.obs_steps) if args.obs_steps is not None else int(cfg.n_obs_steps)

    print(f"checkpoint: {ckpt_path}")
    print(f"policy: {policy.__class__.__name__}")

    i = 0
    while True:
        t_start = time.time()
        i += 1
        obs_dict = build_random_obs(cfg, batch_size=args.batch_size, n_obs_steps=n_obs_steps, device=device)
        with torch.no_grad():
            pred = policy.predict_action(obs_dict)

        print(f"[iter {i}]")
        for key, value in obs_dict.items():
            print(f"obs.{key}.shape = {tuple(value.shape)}")
        print(f"action.shape = {tuple(pred['action'].shape)}")
        print(f"action_pred.shape = {tuple(pred['action_pred'].shape)}")

        if args.once:
            break
        elapsed = time.time() - t_start
        time.sleep(max(0.05 - elapsed, 0.0))


if __name__ == "__main__":
    main()
