import os
import time
import argparse
import threading
from pathlib import Path
from typing import Dict
from collections import deque
import numpy as np

import torch
from omegaconf import OmegaConf

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

# 导入你指定的自定义接口
from rm_ros_interfaces.msg import Jointpos
from my_interfaces.msg import HeaderFloat32

# 导入 DP3 的环境
from train import TrainDP3Workspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

class DP3InferenceNode(Node):
    def __init__(self, cfg, policy, device, n_obs_steps):
        super().__init__('dp3_inference_node')
        self.cfg = cfg
        self.policy = policy
        self.device = device
        self.n_obs_steps = n_obs_steps
        
        # --- 参数配置 ---
        self.control_rate = 20.0  # 默认推理和控制频率修改为 20Hz
        self.inference_rate = 20.0
        self.use_pc_color = bool(OmegaConf.select(cfg, "policy.use_pc_color", default=False))
        self.n_action_steps = int(OmegaConf.select(cfg, "n_action_steps", default=1))
        action_shape = OmegaConf.select(cfg, "shape_meta.action.shape", default=[7])
        self.action_dim = int(action_shape[0]) if len(action_shape) > 0 else 7
        self.max_cached_chunks = int(OmegaConf.select(cfg, "inference.max_cached_chunks", default=3))
        pc_shape = OmegaConf.select(cfg, "shape_meta.obs.point_cloud.shape", default=[1024, 3])
        self.pc_num_points = int(pc_shape[0]) if len(pc_shape) > 0 else 1024
        self.pc_feature_dim = int(pc_shape[1]) if len(pc_shape) > 1 else (6 if self.use_pc_color else 3)
        self.data_lock = threading.Lock()
        self.queue_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.current_action_chunk = None
        self.current_action_idx = 0
        self.action_chunk_queue = deque(maxlen=self.max_cached_chunks)
        
        # --- 观测历史 Buffer (滑动窗口) ---
        self.obs_state_buffer = deque(maxlen=self.n_obs_steps)
        self.obs_pc_buffer = deque(maxlen=self.n_obs_steps)
        
        # --- 订阅者 ---
        self.sub_joint_state = self.create_subscription(
            JointState,
            'right_arm/joint_states', # TODO: 替换为实际关节状态话题
            self.joint_state_cb,
            10
        )
        self.sub_pointcloud = self.create_subscription(
            PointCloud2,
            '/sampled_points', # TODO: 替换为实际降采样后的点云话题
            self.pointcloud_cb,
            10
        )
        
        # --- 发布者 ---
        self.pub_arm_cmd = self.create_publisher(
            Jointpos,
            'right_arm/rm_driver/movej_canfd_cmd', # 机械臂控制话题
            10
        )
        self.pub_gripper_cmd = self.create_publisher(
            HeaderFloat32,
            'right_arm/gripper_cmd',               # 夹爪控制话题
            10
        )
        
        # --- 定时器 (控制循环) ---
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)
        
        # 缓存最新接收到的数据，用于补齐历史
        self.latest_state = None
        self.latest_pc = None
        
        self.get_logger().info(f"DP3 Inference Node Started. Control Rate: {self.control_rate}Hz.")
        self.get_logger().info(f"Observation steps required: {self.n_obs_steps}. Waiting for data...")
        self.get_logger().info(
            f"Point cloud config: use_pc_color={self.use_pc_color}, "
            f"expected_shape=({self.pc_num_points}, {self.pc_feature_dim})"
        )
        self.get_logger().info(
            f"Action config: n_action_steps={self.n_action_steps}, action_dim={self.action_dim}, "
            f"max_cached_chunks={self.max_cached_chunks}"
        )
        self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.inference_thread.start()

    def joint_state_cb(self, msg: JointState):
        # 提取关节位置 (假设前6维是机械臂，如果有夹爪请根据实际情况截取)
        # 这里提取出来的应该与你训练时的 agent_pos 维度和顺序一致
        pos = np.array(msg.position[:6], dtype=np.float32)
        with self.data_lock:
            self.latest_state = pos

    def pointcloud_cb(self, msg: PointCloud2):
        field_names = [f.name for f in msg.fields]
        has_separate_rgb = all(name in field_names for name in ("r", "g", "b"))
        has_packed_rgb = ("rgb" in field_names) or ("rgba" in field_names)

        if self.use_pc_color and (has_separate_rgb or has_packed_rgb):
            if has_separate_rgb:
                points_gen = pc2.read_points(msg, field_names=("x", "y", "z", "r", "g", "b"), skip_nans=True)
                pc_array = np.array(list(points_gen), dtype=np.float32)
                # 常见 ROS 点云颜色是 0-255，这里归一化到 0-1，和训练中常见输入范围对齐
                if pc_array.shape[1] >= 6 and np.max(pc_array[:, 3:6]) > 1.0:
                    pc_array[:, 3:6] = pc_array[:, 3:6] / 255.0
            else:
                packed_name = "rgb" if "rgb" in field_names else "rgba"
                points_gen = pc2.read_points(msg, field_names=("x", "y", "z", packed_name), skip_nans=True)
                raw_array = np.array(list(points_gen))
                if len(raw_array) == 0:
                    return
                xyz = raw_array[:, :3].astype(np.float32)
                rgb = np.stack([self._decode_packed_rgb(v) for v in raw_array[:, 3]], axis=0)
                pc_array = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        else:
            points_gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            xyz_array = np.array(list(points_gen), dtype=np.float32)
            if len(xyz_array) == 0:
                return
            if self.use_pc_color:
                self.get_logger().warn(
                    "PointCloud2 has no color fields (r/g/b/rgb/rgba); fallback to zero color.",
                    throttle_duration_sec=5.0
                )
                zero_color = np.zeros((xyz_array.shape[0], 3), dtype=np.float32)
                pc_array = np.concatenate([xyz_array, zero_color], axis=1)
            else:
                pc_array = xyz_array
        
        if len(pc_array) == 0:
            return

        adapted_pc = self._adapt_point_cloud_shape(pc_array)
        with self.data_lock:
            self.latest_pc = adapted_pc

    @staticmethod
    def _decode_packed_rgb(rgb_value) -> np.ndarray:
        packed = np.array([rgb_value], dtype=np.float32).view(np.uint32)[0]
        r = ((packed >> 16) & 255) / 255.0
        g = ((packed >> 8) & 255) / 255.0
        b = (packed & 255) / 255.0
        return np.array([r, g, b], dtype=np.float32)

    def _adapt_point_cloud_shape(self, pc_array: np.ndarray) -> np.ndarray:
        # 对齐特征维度，避免与训练配置 shape_meta 不一致导致推理崩溃
        cur_feat_dim = pc_array.shape[1]
        if cur_feat_dim > self.pc_feature_dim:
            pc_array = pc_array[:, :self.pc_feature_dim]
        elif cur_feat_dim < self.pc_feature_dim:
            pad = np.zeros((pc_array.shape[0], self.pc_feature_dim - cur_feat_dim), dtype=np.float32)
            pc_array = np.concatenate([pc_array, pad], axis=1)

        # 对齐点数维度
        cur_points = pc_array.shape[0]
        if cur_points > self.pc_num_points:
            idx = np.random.choice(cur_points, self.pc_num_points, replace=False)
            pc_array = pc_array[idx]
        elif cur_points < self.pc_num_points:
            pad = np.zeros((self.pc_num_points - cur_points, self.pc_feature_dim), dtype=np.float32)
            pc_array = np.concatenate([pc_array, pad], axis=0)
        return pc_array.astype(np.float32, copy=False)

    def get_obs_dict(self):
        """
        构建模型输入字典，处理时间维度的滑动窗口
        """
        with self.data_lock:
            if self.latest_state is None or self.latest_pc is None:
                return None
            latest_state = self.latest_state.copy()
            latest_pc = self.latest_pc.copy()

            # 将最新数据压入 buffer
            self.obs_state_buffer.append(latest_state)
            self.obs_pc_buffer.append(latest_pc)

            # 如果刚启动，buffer 没满，用最新数据复制补齐
            while len(self.obs_state_buffer) < self.n_obs_steps:
                self.obs_state_buffer.append(latest_state)
                self.obs_pc_buffer.append(latest_pc)
            
        # 堆叠成 (n_obs_steps, D) 并增加 batch 维度 (1, n_obs_steps, D)
        agent_pos_tensor = torch.tensor(np.stack(self.obs_state_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
        point_cloud_tensor = torch.tensor(np.stack(self.obs_pc_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 注意这里的 key 必须与你训练配置(cfg.shape_meta.obs)里的一致
        obs_dict = {
            'agent_pos': agent_pos_tensor,
            'point_cloud': point_cloud_tensor
        }
        return obs_dict

    def inference_worker(self):
        while not self.stop_event.is_set():
            obs_dict = self.get_obs_dict()
            if obs_dict is None:
                time.sleep(0.01)
                continue

            t_start = time.time()
            try:
                with torch.no_grad():
                    pred = self.policy.predict_action(obs_dict)
                action_seq = pred['action'][0].detach().cpu().numpy()
            except Exception as e:
                self.get_logger().error(f"Inference failed: {e}")
                time.sleep(0.05)
                continue

            if action_seq.ndim != 2 or action_seq.shape[1] != self.action_dim:
                self.get_logger().warn(
                    f"Skip cache: invalid action sequence shape {action_seq.shape}, "
                    f"expected (N, {self.action_dim}).",
                    throttle_duration_sec=2.0
                )
                time.sleep(0.01)
                continue

            if action_seq.shape[0] < self.n_action_steps:
                self.get_logger().warn(
                    f"Skip cache: action sequence too short {action_seq.shape[0]} < {self.n_action_steps}.",
                    throttle_duration_sec=2.0
                )
                time.sleep(0.01)
                continue

            chunk = action_seq[:self.n_action_steps].copy()
            with self.queue_lock:
                if len(self.action_chunk_queue) >= self.max_cached_chunks:
                    # 保留最先要执行的序列，替换最远未来的序列，减少缓存过时动作。
                    self.action_chunk_queue.pop()
                self.action_chunk_queue.append(chunk)

            infer_time = (time.time() - t_start) * 1000
            self.get_logger().debug(f"Inference time: {infer_time:.2f} ms")
            if self.inference_rate > 0:
                time.sleep(max(0.0, 1.0 / self.inference_rate))

    def control_loop(self):
        with self.queue_lock:
            need_new_chunk = (
                self.current_action_chunk is None
                or self.current_action_idx >= len(self.current_action_chunk)
            )
            if need_new_chunk:
                if len(self.action_chunk_queue) == 0:
                    self.current_action_chunk = None
                else:
                    self.current_action_chunk = self.action_chunk_queue.popleft()
                    self.current_action_idx = 0

            if self.current_action_chunk is None:
                target_action = None
            else:
                target_action = self.current_action_chunk[self.current_action_idx]
                self.current_action_idx += 1

        if target_action is None:
            self.get_logger().warn("Action cache is empty. Waiting for inference...", throttle_duration_sec=2.0)
            return

        if target_action.ndim != 1 or target_action.shape[0] != self.action_dim:
            self.get_logger().warn(
                f"Skip publish: invalid action shape {target_action.shape}, expected ({self.action_dim},).",
                throttle_duration_sec=2.0
            )
            return
        if self.action_dim < 7:
            self.get_logger().warn(
                f"Skip publish: action_dim={self.action_dim} < 7 is not supported by current publishers.",
                throttle_duration_sec=2.0
            )
            return

        arm_action = target_action[:6]
        gripper_action = target_action[6]

        self.publish_arm_control(arm_action)
        self.publish_gripper_control(gripper_action)

    def publish_arm_control(self, arm_action):
        msg = Jointpos()
        # 将 numpy 数组转为 list 赋值给 float32[6]
        msg.joint = arm_action.tolist()
        msg.follow = False   # 根据你的文档，默认开启高跟随
        msg.expand = 0.0
        self.pub_arm_cmd.publish(msg)

    def publish_gripper_control(self, gripper_action):
        msg = HeaderFloat32()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "gripper_link"
        msg.data = float(gripper_action)
        self.pub_gripper_cmd.publish(msg)

    def destroy_node(self):
        self.stop_event.set()
        if hasattr(self, "inference_thread") and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        return super().destroy_node()


def find_latest_checkpoint(root: Path) -> Path:
    candidates = list(root.glob("**/checkpoints/latest.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No latest.ckpt found under: {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def main(args=None):
    rclpy.init(args=args)

    # --- 1. 加载模型配置和权重 ---
    default_checkpoint = "3D-Diffusion-Policy/data/outputs/adroit_hammer-simple_dp3-0306_seed0/checkpoints/latest.ckpt"
    ckpt_path = Path(default_checkpoint)
    output_dir = ckpt_path.parent.parent
    cfg_path = output_dir / ".hydra" / "config.yaml"
    
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return

    cfg = OmegaConf.load(cfg_path)
    workspace = TrainDP3Workspace(cfg, output_dir=str(output_dir))
    workspace.load_checkpoint(path=ckpt_path)

    policy = workspace.ema_model if cfg.training.use_ema and workspace.ema_model is not None else workspace.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device).eval()

    n_obs_steps = int(cfg.n_obs_steps)
    print(f"Successfully loaded policy: {policy.__class__.__name__}")

    # --- 2. 启动 ROS2 节点 ---
    node = DP3InferenceNode(cfg=cfg, policy=policy, device=device, n_obs_steps=n_obs_steps)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
