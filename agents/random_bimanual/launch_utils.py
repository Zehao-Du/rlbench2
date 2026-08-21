from helpers.preprocess_agent import PreprocessAgent

from agents.random_bimanual.random_agent import RandomBimanualAgent


def create_agent(cfg):
    agent = RandomBimanualAgent(
        position_noise=cfg.method.position_noise,
        gripper_flip_prob=cfg.method.gripper_flip_prob,
        include_ignore_collisions=cfg.method.get("include_ignore_collisions", True),
        seed=cfg.method.seed,
    )
    return PreprocessAgent(pose_agent=agent, norm_rgb=False)
