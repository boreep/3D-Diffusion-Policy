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
from geometry_msgs.msg import PoseStamped  # 新增：用于发布位姿IK

# 导入你指定的自定义接口
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
        self.control_rate = 20.0  
        self.inference_rate = 20.0
        self.use_pc_color = bool(OmegaConf.select(cfg, "policy.use_pc_color", default=False))
        self.n_action_steps = int(OmegaConf.select(cfg, "n_action_steps", default=1))
        # 默认action shape如果配置没写明，这里假设改为8维
        action_shape = OmegaConf.select(cfg, "shape_meta.action.shape", default=[8])
        self.action_dim = int(action_shape[0]) if len(action_shape) > 0 else 8
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
        
        self.last_state_time = None
        self.last_pc_time = None
        self.last_obs_missing_log_time = 0.0
        
        # --- 观测历史缓冲区 ---
        self.obs_state_buffer = deque(maxlen=self.n_obs_steps)
        self.obs_pc_buffer = deque(maxlen=self.n_obs_steps)
        
        # --- 订阅者 ---
        self.sub_joint_state = self.create_subscription(JointState, 'right_arm/joint_states', self.joint_state_cb, 10)
        self.sub_pointcloud = self.create_subscription(PointCloud2, 'camera/sampled_points', self.pointcloud_cb, 10)
        
        # --- 发布者 ---
        # 修改为发布 PoseStamped 到 right_arm/ik_target_pose
        self.pub_arm_cmd = self.create_publisher(PoseStamped, '/right_arm/ik_target_pose', 10)
        self.pub_gripper_cmd = self.create_publisher(HeaderFloat32, 'right_arm/gripper_cmd', 10)
        
        # --- 定时器 ---
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)
        
        self.latest_state = None
        self.latest_pc = None
        
        self.get_logger().info(f"DP3 Node Started. Control: {self.control_rate}Hz, Actions: {self.n_action_steps} steps.")
        
        self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        self.inference_thread.start()

    def joint_state_cb(self, msg: JointState):
        # 这里的观测假设依然是6维关节角，如果也改成了位姿请相应修改
        pos = np.array(msg.position[:6], dtype=np.float32)
        with self.data_lock:
            self.latest_state = pos
            self.last_state_time = time.time()

    def pointcloud_cb(self, msg: PointCloud2):
        field_names = [f.name for f in msg.fields]
        has_separate_rgb = all(name in field_names for name in ("r", "g", "b"))
        has_packed_rgb = ("rgb" in field_names) or ("rgba" in field_names)

        if self.use_pc_color and (has_separate_rgb or has_packed_rgb):
            if has_separate_rgb:
                points_gen = pc2.read_points(msg, field_names=("x", "y", "z", "r", "g", "b"), skip_nans=True)
                pc_array = self._points_to_float_array(points_gen, ("x", "y", "z", "r", "g", "b"))
                if pc_array.shape[1] >= 6 and np.max(pc_array[:, 3:6]) > 1.0:
                    pc_array[:, 3:6] = pc_array[:, 3:6] / 255.0
            else:
                packed_name = "rgb" if "rgb" in field_names else "rgba"
                points_gen = pc2.read_points(msg, field_names=("x", "y", "z", packed_name), skip_nans=True)
                raw_array = self._points_to_float_array(points_gen, ("x", "y", "z", packed_name))
                if len(raw_array) == 0: return
                xyz = raw_array[:, :3].astype(np.float32)
                rgb = np.stack([self._decode_packed_rgb(v) for v in raw_array[:, 3]], axis=0)
                pc_array = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        else:
            points_gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            xyz_array = self._points_to_float_array(points_gen, ("x", "y", "z"))
            if len(xyz_array) == 0: return
            if self.use_pc_color:
                zero_color = np.zeros((xyz_array.shape[0], 3), dtype=np.float32)
                pc_array = np.concatenate([xyz_array, zero_color], axis=1)
            else:
                pc_array = xyz_array
        
        adapted_pc = self._adapt_point_cloud_shape(pc_array)
        with self.data_lock:
            self.latest_pc = adapted_pc
            self.last_pc_time = time.time()

    @staticmethod
    def _decode_packed_rgb(rgb_value) -> np.ndarray:
        packed = np.array([rgb_value], dtype=np.float32).view(np.uint32)[0]
        r, g, b = ((packed >> 16) & 255) / 255.0, ((packed >> 8) & 255) / 255.0, (packed & 255) / 255.0
        return np.array([r, g, b], dtype=np.float32)

    @staticmethod
    def _points_to_float_array(points_gen, ordered_names) -> np.ndarray:
        raw = np.array(list(points_gen))
        if raw.size == 0: return np.zeros((0, len(ordered_names)), dtype=np.float32)
        if raw.dtype.names is not None:
            cols = [raw[name].astype(np.float32) for name in ordered_names]
            return np.stack(cols, axis=1)
        arr = np.asarray(raw, dtype=np.float32)
        return arr.reshape(-1, len(ordered_names)) if arr.ndim == 1 else arr

    def _adapt_point_cloud_shape(self, pc_array: np.ndarray) -> np.ndarray:
        cur_feat_dim = pc_array.shape[1]
        if cur_feat_dim > self.pc_feature_dim:
            pc_array = pc_array[:, :self.pc_feature_dim]
        elif cur_feat_dim < self.pc_feature_dim:
            pad = np.zeros((pc_array.shape[0], self.pc_feature_dim - cur_feat_dim), dtype=np.float32)
            pc_array = np.concatenate([pc_array, pad], axis=1)

        cur_points = pc_array.shape[0]
        if cur_points > self.pc_num_points:
            idx = np.random.choice(cur_points, self.pc_num_points, replace=False)
            pc_array = pc_array[idx]
        elif cur_points < self.pc_num_points:
            pad = np.zeros((self.pc_num_points - cur_points, self.pc_feature_dim), dtype=np.float32)
            pc_array = np.concatenate([pc_array, pad], axis=0)
        return pc_array.astype(np.float32, copy=False)

    def get_obs_dict(self):
        with self.data_lock:
            if self.latest_state is None or self.latest_pc is None:
                now = time.time()
                if now - self.last_obs_missing_log_time > 5.0:
                    self.get_logger().warn("Waiting for state and pointcloud data...")
                    self.last_obs_missing_log_time = now
                return None
            
            latest_state, latest_pc = self.latest_state.copy(), self.latest_pc.copy()
            self.obs_state_buffer.append(latest_state)
            self.obs_pc_buffer.append(latest_pc)

            while len(self.obs_state_buffer) < self.n_obs_steps:
                self.obs_state_buffer.append(latest_state)
                self.obs_pc_buffer.append(latest_pc)
            
        return {
            'agent_pos': torch.tensor(np.stack(self.obs_state_buffer), dtype=torch.float32).unsqueeze(0).to(self.device),
            'point_cloud': torch.tensor(np.stack(self.obs_pc_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
        }

    def inference_worker(self):
        while not self.stop_event.is_set():
            obs_dict = self.get_obs_dict()
            if obs_dict is None:
                time.sleep(0.01)
                continue

            try:
                with torch.no_grad():
                    pred = self.policy.predict_action(obs_dict)
                action_seq = pred['action'][0].detach().cpu().numpy()
                
                if action_seq.shape[0] >= self.n_action_steps:
                    chunk = action_seq[:self.n_action_steps].copy()
                    with self.queue_lock:
                        if len(self.action_chunk_queue) >= self.max_cached_chunks:
                            self.action_chunk_queue.popleft()
                        self.action_chunk_queue.append(chunk)
            except Exception as e:
                self.get_logger().error(f"Inference failed: {e}", throttle_duration_sec=5.0)
                time.sleep(0.05)

            if self.inference_rate > 0:
                time.sleep(max(0.0, 1.0 / self.inference_rate))

    def control_loop(self):
        with self.queue_lock:
            if self.current_action_chunk is None or self.current_action_idx >= len(self.current_action_chunk):
                if len(self.action_chunk_queue) > 0:
                    self.current_action_chunk = self.action_chunk_queue.popleft()
                    self.current_action_idx = 0
                else:
                    self.current_action_chunk = None

            if self.current_action_chunk is not None:
                target_action = self.current_action_chunk[self.current_action_idx]
                self.current_action_idx += 1
                
                # 发布控制：前7维给IK位姿，第8维给夹爪
                self.publish_arm_control(target_action[:7])
                self.publish_gripper_control(target_action[7])
            else:
                self.get_logger().debug("Action cache empty", throttle_duration_sec=2.0)

    def publish_arm_control(self, ik):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        # 注意：这里的 frame_id 使用了你要求的 "ee_link"，在实际机器人中，IK目标位姿通常是相对于基座坐标系的（例如 "base_link"），如有需要你可以自行调整
        msg.header.frame_id = "ee_link"

        msg.pose.position.x = float(ik[0])
        msg.pose.position.y = float(ik[1])
        msg.pose.position.z = float(ik[2])

        msg.pose.orientation.x = float(ik[3])  # qx
        msg.pose.orientation.y = float(ik[4])  # qy
        msg.pose.orientation.z = float(ik[5])  # qz
        msg.pose.orientation.w = float(ik[6])  # qw

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
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    ckpt_path = Path("3D-Diffusion-Policy/data/outputs/real_fruit-real_simple_dp3-0308_seed0/checkpoints/epoch=1000-test_mean_score=-0.000.ckpt")
    cfg_path = ckpt_path.parent.parent / ".hydra" / "config.yaml"
    
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return

    cfg = OmegaConf.load(cfg_path)
    workspace = TrainDP3Workspace(cfg, output_dir=str(ckpt_path.parent.parent))
    workspace.load_checkpoint(path=ckpt_path)

    policy = workspace.ema_model if cfg.training.use_ema and workspace.ema_model is not None else workspace.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device).eval()

    node = DP3InferenceNode(cfg=cfg, policy=policy, device=device, n_obs_steps=int(cfg.n_obs_steps))

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()