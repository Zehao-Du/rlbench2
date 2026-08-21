from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from yarr.agents.agent import ActResult, Agent, Summary


class Map4DDiTRLBenchAgent(Agent):
    def __init__(
        self,
        actor_network: nn.Module,
        cameras: List[str],
        cameras_pcd: List[str],
        episode_length: int,
        query_freq: int,
        action_horizon: int,
        n_obs_steps: int,
        fps_num: int,
        bounding_box,
        semantic_feature_source: str,
        dino_feature_dim: int,
        dinov2_repo_path: str,
        dinov2_weights_path: str,
        ckpt_path: str,
        map_pose_source: str = "simulator",
        map_object_name: str = "cube",
        use_ema: bool = True,
        strict_load: bool = True,
        debug_map_video_path: str = "",
        debug_map_video_camera: str = "front",
        debug_map_video_fps: int = 10,
        device_cfg=0,
    ):
        self._actor = actor_network
        self.cameras = list(cameras)
        self.cameras_pcd = list(cameras_pcd)
        self._episode_length = int(episode_length)
        self.query_freq = int(query_freq)
        self.action_horizon = int(action_horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.fps_num = int(fps_num)
        self.bounding_box = bounding_box
        self.semantic_feature_source = str(semantic_feature_source)
        self.dino_feature_dim = int(dino_feature_dim)
        self.dinov2_repo_path = str(dinov2_repo_path)
        self.dinov2_weights_path = str(dinov2_weights_path)
        self.map_pose_source = str(map_pose_source)
        self.map_object_name = str(map_object_name)
        self.ckpt_path = str(ckpt_path)
        self.use_ema = bool(use_ema)
        self.strict_load = bool(strict_load)
        self.debug_map_video_path = str(debug_map_video_path or "")
        self.debug_map_video_camera = str(debug_map_video_camera)
        self.debug_map_video_fps = int(debug_map_video_fps)
        self.device_cfg = device_cfg
        self._history = deque(maxlen=self.n_obs_steps)
        self._cached_action = None
        self._action_id = 0
        self._timestep = 0
        self._fusion = None
        self._debug_video_writer = None
        self._map_shape = None

    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device("cuda", int(self.device_cfg)) if torch.cuda.is_available() else torch.device("cpu")
        self._device = device
        self._actor = self._actor.to(device).train(training)
        if self.ckpt_path:
            self._load_checkpoint(Path(self.ckpt_path), device)
        if self.semantic_feature_source != "fusion":
            raise ValueError(
                "RLBench2 Map4D DiT eval requires semantic_feature_source='fusion' "
                "to compute real DINO features. RGB/zero/none semantic fallbacks are not allowed."
            )
        from map4d.backbone.model.vision.semantic_feature_extractor import Fusion

        self._fusion = Fusion(
            num_cam=len(self.cameras),
            feat_backbone="dinov2",
            device=device,
            dinov2_repo_path=self.dinov2_repo_path,
            dinov2_weights_path=self.dinov2_weights_path,
        )
        if self.debug_map_video_path:
            import imageio.v2 as imageio

            video_path = Path(self.debug_map_video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            self._debug_video_writer = imageio.get_writer(
                str(video_path),
                fps=self.debug_map_video_fps,
                codec="libx264",
                quality=8,
            )
        if self.map_pose_source != "simulator":
            raise ValueError(
                f"Unsupported map_pose_source={self.map_pose_source!r}; "
                "RLBench2 Map4D DiT eval requires 'simulator'. Point-cloud bbox fallback is not allowed."
            )
        self._map_shape = None

    def reset(self):
        self._history.clear()
        self._cached_action = None
        self._action_id = 0
        self._timestep = 0

    def act(self, step: int, observation: dict, deterministic=False) -> ActResult:
        obs_frame = self._frame_to_model_obs(observation)
        self._write_debug_map_frame(observation, obs_frame)
        self._history.append(obs_frame)
        while len(self._history) < self.n_obs_steps:
            self._history.appendleft(obs_frame)

        model_obs = self._stack_history()
        if self._cached_action is None or self._timestep % self.query_freq == 0:
            with torch.no_grad():
                result = self._actor.predict_action(model_obs)
            self._cached_action = result["action"].detach()
            self._action_id = 0

        action = self._select_action(self._cached_action, self._action_id)
        if self._action_id < max(0, self.action_horizon - 1):
            self._action_id += 1
        self._timestep += 1
        if self._timestep >= self._episode_length:
            self._close_debug_video()
        return ActResult(action.detach().cpu().numpy())

    def load_weights(self, savedir: str):
        if self.ckpt_path:
            self._load_checkpoint(Path(self.ckpt_path), self._device)

    def save_weights(self, savedir: str):
        return

    def update(self, step: int, replay_sample: dict) -> dict:
        raise NotImplementedError("MAP4D_DIT RLBench2 agent is evaluation-only")

    def update_summaries(self) -> List[Summary]:
        return []

    def act_summaries(self) -> List[Summary]:
        return []

    def __del__(self):
        self._close_debug_video()

    def _load_checkpoint(self, ckpt_path: Path, device: torch.device):
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Map4D DiT checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        state_key = "ema_model_state_dict" if self.use_ema and checkpoint.get("ema_model_state_dict") is not None else "model_state_dict"
        state_dict = checkpoint.get(state_key, checkpoint)
        self._actor.load_state_dict(state_dict, strict=self.strict_load)
        self._actor.eval()
        print(f"Loaded MAP4D_DIT weights from {ckpt_path} ({state_key})")

    def _frame_to_model_obs(self, observation: dict):
        left_pose = observation["left_gripper_pose"].to(self._device).reshape(-1)
        right_pose = observation["right_gripper_pose"].to(self._device).reshape(-1)
        left_open = observation["left_gripper_open"].to(self._device).reshape(-1)[:1]
        right_open = observation["right_gripper_open"].to(self._device).reshape(-1)[:1]
        robot_state = torch.cat([left_pose, right_pose, left_open, right_open], dim=-1).float()

        point_cloud = self._sample_point_cloud(observation)
        dino_feature = self._dino_feature(observation, point_cloud)
        node_poses, size_parameters, relation_parameters = self._push_box_map(point_cloud)
        return {
            "robot_state": robot_state,
            "point_cloud": point_cloud,
            "dino_feature": dino_feature,
            "node_poses": node_poses,
            "size_parameters": size_parameters,
            "relation_parameters": relation_parameters,
        }

    def _stack_history(self):
        keys = self._history[0].keys()
        stacked = {}
        for key in keys:
            values = [frame[key] for frame in self._history]
            if key in {"size_parameters", "relation_parameters"}:
                stacked[key] = values[-1].unsqueeze(0)
            else:
                stacked[key] = torch.stack(values, dim=0).unsqueeze(0)
        return stacked

    def _sample_point_cloud(self, observation: dict) -> torch.Tensor:
        pcs = []
        rgbs = []
        for camera in self.cameras_pcd:
            pc = observation[f"{camera}_point_cloud"].to(self._device).float()
            rgb = observation.get(f"{camera}_rgb")
            if rgb is None:
                raise KeyError(f"Observation is missing required RGB image for camera {camera!r}")
            pc = self._squeeze_obs_image(pc)
            rgb = self._squeeze_obs_image(rgb.to(self._device).float())
            if rgb.shape[0] == 3:
                rgb = rgb.permute(1, 2, 0)
            rgb = self._rgb_to_unit(rgb).reshape(-1, 3)
            pc = pc.permute(1, 2, 0).reshape(-1, 3) if pc.shape[0] == 3 else pc.reshape(-1, 3)
            pcs.append(pc)
            rgbs.append(rgb)
        xyz = torch.cat(pcs, dim=0)
        color = torch.cat(rgbs, dim=0)
        full = torch.cat([xyz, color], dim=-1)
        return self._random_sample_in_bbox(full, self.fps_num)

    def _dino_feature(self, observation: dict, point_cloud: torch.Tensor) -> torch.Tensor:
        if self.semantic_feature_source != "fusion":
            raise ValueError(
                "Unsupported semantic_feature_source="
                f"{self.semantic_feature_source!r}; RLBench2 Map4D DiT requires 'fusion'."
            )
        if self._fusion is None:
            raise RuntimeError("semantic_feature_source=fusion but Fusion was not initialized")
        images, depths, extrinsics, intrinsics = [], [], [], []
        for camera in self.cameras:
            rgb = self._squeeze_obs_image(observation[f"{camera}_rgb"].to(self._device).float())
            depth = self._squeeze_obs_image(observation[f"{camera}_depth"].to(self._device).float())
            if rgb.shape[0] == 3:
                rgb = rgb.permute(1, 2, 0)
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]
            elif depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            images.append(self._rgb_to_255(rgb).clamp(0, 255).to(torch.uint8).detach().cpu().numpy())
            depths.append(depth.detach().cpu().numpy())
            extrinsics.append(self._transform_camera_extrinsics(
                self._squeeze_matrix(observation[f"{camera}_camera_extrinsics"].to(self._device).float())
            ).detach().cpu().numpy())
            intrinsics.append(
                self._squeeze_matrix(observation[f"{camera}_camera_intrinsics"].to(self._device).float())
                .detach()
                .cpu()
                .numpy()
            )
        obs = {
            "color": np.stack(images, axis=0),
            "depth": np.stack(depths, axis=0),
            "pose": np.stack(extrinsics, axis=0),
            "K": np.stack(intrinsics, axis=0),
        }
        feat = self._fusion.extract_semantic_feature_from_ptc(point_cloud[:, :3], obs)
        return feat.reshape(self.fps_num, -1).to(self._device)

    def _push_box_map(self, point_cloud: torch.Tensor):
        if self.map_pose_source == "simulator":
            return self._simulator_push_box_map(point_cloud)
        raise ValueError(
            f"Unsupported map_pose_source={self.map_pose_source!r}; "
            "RLBench2 Map4D DiT requires simulator pose."
        )

    def _simulator_push_box_map(self, point_cloud: torch.Tensor):
        if self._map_shape is None:
            from pyrep.objects.shape import Shape

            self._map_shape = Shape(self.map_object_name)
        pose_xyzw = torch.as_tensor(
            self._map_shape.get_pose(),
            dtype=point_cloud.dtype,
            device=point_cloud.device,
        )
        quat_xyzw = pose_xyzw[3:7]
        quat_wxyz = torch.cat([quat_xyzw[3:4], quat_xyzw[0:3]], dim=0)
        quat_wxyz = quat_wxyz / quat_wxyz.norm().clamp_min(1e-8)
        node_pose = torch.cat([pose_xyzw[0:3], quat_wxyz], dim=0).reshape(1, 7)

        bbox = self._map_shape.get_bounding_box()
        size = torch.tensor(
            [bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]],
            dtype=point_cloud.dtype,
            device=point_cloud.device,
        ).clamp_min(0.01)
        return node_pose, size, point_cloud.new_zeros(0)

    def _write_debug_map_frame(self, observation: dict, obs_frame: dict) -> None:
        if self._debug_video_writer is None:
            return
        import cv2

        camera = self.debug_map_video_camera
        rgb_key = f"{camera}_rgb"
        intrinsics_key = f"{camera}_camera_intrinsics"
        extrinsics_key = f"{camera}_camera_extrinsics"
        missing = [key for key in (rgb_key, intrinsics_key, extrinsics_key) if key not in observation]
        if missing:
            raise KeyError(f"Cannot draw Map4D debug video; observation is missing {missing}")

        rgb = self._squeeze_obs_image(observation[rgb_key].to(self._device).float())
        if rgb.shape[0] == 3:
            rgb = rgb.permute(1, 2, 0)
        image = (self._rgb_to_unit(rgb).detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        draw = np.ascontiguousarray(image.copy())

        intrinsics = self._squeeze_matrix(observation[intrinsics_key].to(self._device).float())
        extrinsics = self._squeeze_matrix(observation[extrinsics_key].to(self._device).float())
        center = obs_frame["node_poses"][0, 0:3]
        quat_wxyz = obs_frame["node_poses"][0, 3:7]
        extent = obs_frame["size_parameters"]
        corners = self._box_corners(center, extent, quat_wxyz)
        projected_corners = self._project_world_points(corners, intrinsics, extrinsics)
        projected_center = self._project_world_points(center.reshape(1, 3), intrinsics, extrinsics)

        edges = (
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        for a, b in edges:
            pa = projected_corners[a]
            pb = projected_corners[b]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                cv2.line(draw, tuple(pa.astype(int)), tuple(pb.astype(int)), (0, 255, 255), 2, lineType=cv2.LINE_AA)

        pc = projected_center[0]
        if np.isfinite(pc).all():
            cv2.circle(draw, tuple(pc.astype(int)), 6, (255, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                draw,
                "Map4D node",
                tuple((pc + np.array([8, -8])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )
        label = (
            f"step={self._timestep} "
            f"center=({center[0].item():.3f},{center[1].item():.3f},{center[2].item():.3f}) "
            f"size=({extent[0].item():.3f},{extent[1].item():.3f},{extent[2].item():.3f})"
        )
        cv2.putText(draw, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(draw, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        self._debug_video_writer.append_data(draw)

    @staticmethod
    def _box_corners(center: torch.Tensor, extent: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
        offsets = torch.tensor(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=center.dtype,
            device=center.device,
        )
        rotation = Map4DDiTRLBenchAgent._quat_wxyz_to_matrix(quat_wxyz)
        local = offsets * extent.reshape(1, 3)
        return local @ rotation.T + center.reshape(1, 3)

    @staticmethod
    def _quat_wxyz_to_matrix(quat_wxyz: torch.Tensor) -> torch.Tensor:
        quat = quat_wxyz / quat_wxyz.norm().clamp_min(1e-8)
        w, x, y, z = quat.unbind(dim=-1)
        return torch.stack(
            [
                torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
                torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
                torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
            ],
            dim=0,
        )

    @staticmethod
    def _project_world_points(points: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> np.ndarray:
        if intrinsics.shape != (3, 3):
            raise ValueError(f"camera intrinsics must be [3,3], got {tuple(intrinsics.shape)}")
        if extrinsics.shape != (4, 4):
            raise ValueError(f"camera extrinsics must be [4,4], got {tuple(extrinsics.shape)}")
        points_h = torch.cat([points, torch.ones(points.shape[0], 1, dtype=points.dtype, device=points.device)], dim=-1)
        world_to_camera = torch.linalg.inv(extrinsics)
        camera_points = points_h @ world_to_camera.T
        z = camera_points[:, 2].clamp_min(1e-6)
        uvw = camera_points[:, :3] @ intrinsics.T
        uv = uvw[:, :2] / z[:, None]
        valid = camera_points[:, 2] > 1e-6
        uv = uv.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        uv[~valid_np] = np.nan
        return uv

    def _close_debug_video(self) -> None:
        writer = self._debug_video_writer
        self._debug_video_writer = None
        if writer is not None:
            writer.close()

    def _random_sample_in_bbox(self, pc: torch.Tensor, num_samples: int) -> torch.Tensor:
        bbox = torch.as_tensor(self.bounding_box, dtype=pc.dtype, device=pc.device)
        mask = ((pc[:, :3] >= bbox[0]) & (pc[:, :3] <= bbox[1])).all(dim=-1)
        inside = pc[mask]
        if inside.shape[0] == 0:
            inside = pc
        replace = inside.shape[0] < num_samples
        idx = torch.randint(inside.shape[0], (num_samples,), device=pc.device) if replace else torch.randperm(inside.shape[0], device=pc.device)[:num_samples]
        return inside[idx]

    @staticmethod
    def _squeeze_obs_image(value: torch.Tensor) -> torch.Tensor:
        while value.ndim > 3:
            value = value[0]
        return value

    @staticmethod
    def _squeeze_matrix(value: torch.Tensor) -> torch.Tensor:
        while value.ndim > 2:
            value = value[0]
        return value

    @staticmethod
    def _rgb_to_255(rgb: torch.Tensor) -> torch.Tensor:
        if rgb.numel() == 0:
            return rgb
        if float(rgb.max().detach().cpu()) <= 2.0:
            return torch.round((rgb + 1.0) * 127.5)
        return torch.round(rgb)

    @staticmethod
    def _rgb_to_unit(rgb: torch.Tensor) -> torch.Tensor:
        if rgb.numel() == 0:
            return rgb
        rgb_min = float(rgb.min().detach().cpu())
        rgb_max = float(rgb.max().detach().cpu())
        if rgb_min < 0.0:
            return ((rgb + 1.0) * 0.5).clamp(0.0, 1.0)
        if rgb_max > 2.0:
            return (rgb / 255.0).clamp(0.0, 1.0)
        return rgb.clamp(0.0, 1.0)

    @staticmethod
    def _transform_camera_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
        if extrinsics.shape[-2:] == (4, 4):
            extrinsics = extrinsics[:3]
        c = extrinsics[:3, 3:]
        r = extrinsics[:3, :3]
        r_inv = r.transpose(-1, -2)
        r_inv_c = torch.matmul(r_inv, c)
        return torch.cat((r_inv, -r_inv_c), dim=-1)

    @staticmethod
    def _select_action(action: torch.Tensor, action_id: int) -> torch.Tensor:
        action = action[0]
        if action.ndim == 3:
            left = action[0, action_id]
            right = action[1, action_id]
        else:
            left = action[action_id, :8]
            right = action[action_id, 8:16]
        left_pose = Map4DDiTRLBenchAgent._model_pose_to_rlbench(left, arm_name="left")
        right_pose = Map4DDiTRLBenchAgent._model_pose_to_rlbench(right, arm_name="right")
        left_open = left[7:8].clamp(0.0, 1.0)
        right_open = right[7:8].clamp(0.0, 1.0)
        one = torch.ones_like(left_open)
        return torch.cat([right_pose, right_open, one, left_pose, left_open, one], dim=-1)

    @staticmethod
    def _model_pose_to_rlbench(arm_action: torch.Tensor, *, arm_name: str) -> torch.Tensor:
        if arm_action.shape != (8,):
            raise ValueError(f"{arm_name} model action must have shape (8,), got {tuple(arm_action.shape)}")
        if not torch.isfinite(arm_action).all():
            raise ValueError(f"{arm_name} model action contains non-finite values")
        quat_wxyz = arm_action[3:7]
        quat_norm = quat_wxyz.norm()
        if float(quat_norm.detach().cpu()) <= 1e-8:
            raise ValueError(f"{arm_name} model action has a zero-length WXYZ quaternion")
        quat_wxyz = quat_wxyz / quat_norm
        # Map4D is trained with canonical WXYZ quaternions. PyRep/RLBench
        # accepts end-effector poses as XYZW, so convert at the API boundary.
        quat_xyzw = torch.cat([quat_wxyz[1:4], quat_wxyz[0:1]], dim=0)
        return torch.cat([arm_action[:3], quat_xyzw], dim=0)
