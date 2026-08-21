from __future__ import annotations

import gc
import logging
import os
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]
DEFAULT_COPPELIASIM_ROOT = PROJECT_ROOT / "codes" / "CoppeliaSim"
_configured_coppelia = pathlib.Path(os.environ.get("COPPELIASIM_ROOT", ""))
if not (
    (_configured_coppelia / "libcoppeliaSim.so.1").is_file()
    and (_configured_coppelia / "platforms" / "libqxcb.so").is_file()
):
    os.environ["COPPELIASIM_ROOT"] = str(DEFAULT_COPPELIASIM_ROOT)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]
_ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
_required_ld_paths = [os.environ["COPPELIASIM_ROOT"]]
if os.environ.get("CONDA_PREFIX"):
    _required_ld_paths.append(str(pathlib.Path(os.environ["CONDA_PREFIX"]) / "lib"))
if (
    any(path not in _ld_paths for path in _required_ld_paths)
    and os.environ.get("MAP4D_RLBENCH_LD_REEXEC") != "1"
):
    env = os.environ.copy()
    prefix = os.pathsep.join(_required_ld_paths)
    env["LD_LIBRARY_PATH"] = prefix + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["MAP4D_RLBENCH_LD_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(_required_ld_paths) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

import hydra
import peract_config
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm
from rlbench.action_modes.action_mode import (
    BimanualJointPositionActionMode,
    BimanualMoveArmThenGripper,
)
from rlbench.action_modes.arm_action_modes import (
    BimanualEndEffectorPoseViaPlanning,
    BimanualJointPosition,
)
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from yarr.runners.independent_env_runner import IndependentEnvRunner
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import SimpleAccumulator

from agents import agent_factory
from helpers import observation_utils, utils


class ProgressRolloutGenerator(RolloutGenerator):
    def generator(
        self,
        step_signal,
        env,
        agent,
        episode_length,
        timesteps,
        eval,
        eval_demo_seed=0,
        record_enabled=False,
    ):
        rollout = super().generator(
            step_signal,
            env,
            agent,
            episode_length,
            timesteps,
            eval,
            eval_demo_seed=eval_demo_seed,
            record_enabled=record_enabled,
        )
        yield from tqdm(
            rollout,
            total=episode_length,
            desc=f"Eval seed {eval_demo_seed}",
            unit="step",
            dynamic_ncols=True,
            smoothing=0.0,
        )


def _normalise_demo_path(demo_path: str, tasks: list[str], logdir: str) -> str:
    """Accept either RLBench dataset roots or a directly extracted task folder."""
    if any(
        os.path.isdir(os.path.join(demo_path, task, "all_variations", "episodes"))
        for task in tasks
    ):
        return demo_path

    if (
        len(tasks) == 1
        and os.path.isdir(os.path.join(demo_path, "all_variations", "episodes"))
    ):
        view_root = os.path.join(logdir, "demo_view")
        os.makedirs(view_root, exist_ok=True)
        link_path = os.path.join(view_root, tasks[0])
        target_path = os.path.abspath(demo_path)
        if os.path.islink(link_path):
            if os.path.realpath(link_path) != target_path:
                os.unlink(link_path)
                os.symlink(target_path, link_path)
        elif not os.path.exists(link_path):
            os.symlink(target_path, link_path)
        else:
            raise RuntimeError(
                f"Cannot create RLBench2 demo view at {link_path}: path already exists"
            )
        return view_root

    return demo_path


def eval_seed(eval_cfg, weightsdir, logdir, env_device, multi_task, env_config) -> None:
    agent = agent_factory.create_agent(eval_cfg)
    stat_accum = SimpleAccumulator(eval_video_fps=30)
    env_runner = IndependentEnvRunner(
        train_env=None,
        agent=agent,
        train_replay_buffer=None,
        num_train_envs=0,
        num_eval_envs=eval_cfg.framework.eval_envs,
        rollout_episodes=99999,
        eval_episodes=eval_cfg.framework.eval_episodes,
        training_iterations=eval_cfg.framework.training_iterations,
        eval_from_eps_number=eval_cfg.framework.eval_from_eps_number,
        episode_length=eval_cfg.rlbench.episode_length,
        stat_accumulator=stat_accum,
        weightsdir=weightsdir,
        logdir=logdir,
        env_device=env_device,
        rollout_generator=ProgressRolloutGenerator(),
        num_eval_runs=len(eval_cfg.rlbench.tasks),
        multi_task=multi_task,
    )
    env_runner._on_thread_start = peract_config.config_logging
    manager = mp.Manager()
    save_load_lock = manager.Lock()
    writer_lock = manager.Lock()
    env_runner.start(
        int(eval_cfg.framework.eval_type),
        save_load_lock,
        writer_lock,
        env_config,
        0,
        eval_cfg.framework.eval_save_metrics,
        eval_cfg.cinematic_recorder,
    )
    del env_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(config_name="eval_map4d_dit", config_path="conf", version_base=None)
def main(eval_cfg: DictConfig) -> None:
    logging.info("\n" + OmegaConf.to_yaml(eval_cfg))
    if eval_cfg.method.name == "MAP4D_DIT" and not eval_cfg.framework.ckpt_path:
        raise ValueError("Set MAP4D_DIT_CKPT or framework.ckpt_path to a Map4D DiT checkpoint")
    if not os.path.isdir(eval_cfg.rlbench.demo_path):
        raise FileNotFoundError(
            f"RLBench2 demo_path is not a directory: {eval_cfg.rlbench.demo_path}. "
            "Mount or extract bimanual_push_box.test.squashfs and set RLBENCH2_TEST_DEMO_PATH."
        )

    logdir = os.path.join(
        eval_cfg.framework.logdir,
        eval_cfg.rlbench.task_name,
        eval_cfg.method.name,
        f"seed{eval_cfg.framework.start_seed}",
    )
    weightsdir = eval_cfg.framework.weightsdir or os.path.dirname(eval_cfg.framework.ckpt_path)
    if eval_cfg.method.name == "RANDOM_BIMANUAL":
        weightsdir = weightsdir or os.path.join(logdir, "dummy_weights")
        os.makedirs(weightsdir, exist_ok=True)
    demo_path = _normalise_demo_path(
        eval_cfg.rlbench.demo_path, list(eval_cfg.rlbench.tasks), logdir
    )
    env_device = utils.get_device(eval_cfg.framework.gpu)
    gripper_mode = eval(eval_cfg.rlbench.gripper_mode)()
    arm_action_mode = eval(eval_cfg.rlbench.arm_action_mode)()
    action_mode = eval(eval_cfg.rlbench.action_mode)(arm_action_mode, gripper_mode)

    task_path = rlbench_task.BIMANUAL_TASKS_PATH
    task_files = [
        task.replace(".py", "")
        for task in os.listdir(task_path)
        if task != "__init__.py" and task.endswith(".py")
    ]
    eval_cfg.rlbench.cameras = (
        eval_cfg.rlbench.cameras
        if isinstance(eval_cfg.rlbench.cameras, ListConfig)
        else [eval_cfg.rlbench.cameras]
    )
    obs_config = observation_utils.create_obs_config(
        eval_cfg.rlbench.cameras,
        eval_cfg.rlbench.camera_resolution,
        eval_cfg.method.name,
        eval_cfg.method.robot_name,
    )
    if eval_cfg.cinematic_recorder.enabled:
        obs_config.record_gripper_closing = True

    task_classes = []
    for task in eval_cfg.rlbench.tasks:
        if task not in task_files:
            raise ValueError(f"Task {task} not recognised")
        task_classes.append(task_file_to_task_class(task, True))

    multi_task = len(task_classes) > 1
    if multi_task:
        env_config = (
            task_classes,
            obs_config,
            action_mode,
            demo_path,
            eval_cfg.rlbench.episode_length,
            eval_cfg.rlbench.headless,
            eval_cfg.framework.eval_episodes,
            eval_cfg.rlbench.include_lang_goal_in_obs,
            eval_cfg.rlbench.time_in_state,
            eval_cfg.framework.record_every_n,
        )
    else:
        env_config = (
            task_classes[0],
            obs_config,
            action_mode,
            demo_path,
            eval_cfg.rlbench.episode_length,
            eval_cfg.rlbench.headless,
            eval_cfg.rlbench.include_lang_goal_in_obs,
            eval_cfg.rlbench.time_in_state,
            eval_cfg.framework.record_every_n,
        )
    eval_seed(eval_cfg, weightsdir, logdir, env_device, multi_task, env_config)


if __name__ == "__main__":
    peract_config.on_init()
    main()
