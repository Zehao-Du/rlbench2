from pathlib import Path
import sys

import hydra
from helpers.preprocess_agent import PreprocessAgent
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.map4d_dit.map4d_agent import Map4DDiTRLBenchAgent  # noqa: E402


def create_agent(cfg: DictConfig):
    policy_cfg = OmegaConf.create(OmegaConf.to_container(cfg.method.policy, resolve=True))
    actor_net = hydra.utils.instantiate(policy_cfg)
    agent = Map4DDiTRLBenchAgent(
        actor_network=actor_net,
        cameras=cfg.rlbench.cameras,
        cameras_pcd=cfg.rlbench.get("cameras_pcd", cfg.rlbench.cameras),
        episode_length=cfg.rlbench.episode_length,
        query_freq=cfg.rlbench.query_freq,
        action_horizon=cfg.method.action_horizon,
        n_obs_steps=cfg.method.n_obs_steps,
        fps_num=cfg.method.fps_num,
        bounding_box=cfg.method.bounding_box,
        semantic_feature_source=cfg.method.semantic_feature_source,
        dino_feature_dim=cfg.method.dino_feature_dim,
        dinov2_repo_path=cfg.method.dinov2_repo_path,
        dinov2_weights_path=cfg.method.dinov2_weights_path,
        ckpt_path=cfg.framework.ckpt_path,
        map_pose_source=cfg.method.get("map_pose_source", "simulator"),
        map_object_name=cfg.method.get("map_object_name", "cube"),
        use_ema=cfg.framework.get("use_ema", True),
        strict_load=cfg.framework.get("strict_load", True),
        debug_map_video_path=cfg.method.get("debug_map_video_path", ""),
        debug_map_video_camera=cfg.method.get("debug_map_video_camera", "front"),
        debug_map_video_fps=cfg.method.get("debug_map_video_fps", 10),
        device_cfg=OmegaConf.select(cfg, "framework.gpu", default=0),
    )
    return PreprocessAgent(pose_agent=agent, norm_rgb=False)
