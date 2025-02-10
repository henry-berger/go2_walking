import isaacgym

assert isaacgym
import torch
import numpy as np
import time

import glob
import pickle as pkl

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Float32MultiArray
import tf

from go2_gym.envs import *
from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.go2.go2_config import config_go2
from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm


def load_policy(logdir):
    body = torch.jit.load(logdir + "/checkpoints/body_latest.jit")
    import os

    adaptation_module = torch.jit.load(
        logdir + "/checkpoints/adaptation_module_latest.jit"
    )

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to("cpu"))
        action = body.forward(torch.cat((obs["obs_history"].to("cpu"), latent), dim=-1))
        info["latent"] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", "rb") as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    # default control_typw is "actuator_net", you can also switch it to "P" to enable joint PD control
    Cfg.control.control_type = "actuator_net"
    Cfg.asset.flip_visual_attachments = True

    from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device="cuda:0", headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go2_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


def play_go2(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go2_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    rospy.init_node("state_publisher", anonymous=True)
    br = tf.TransformBroadcaster()
    pub = rospy.Publisher("joint_states", JointState, queue_size=10)
    sensor_pub = rospy.Publisher("sensor_data", Float32MultiArray, queue_size=10)
    name = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]
    velocity = []
    effort = []

    # label = "gait-conditioned-agility/pretrain-v0/train"
    label = "gait-conditioned-agility/pretrain-go2/train"

    env, policy = load_env(label, headless=headless)

    num_eval_steps = 2500  # 250
    gaits = {
        "pronking": [0, 0, 0],
        "trotting": [0.5, 0, 0],
        "bounding": [0, 0.5, 0],
        "pacing": [0, 0, 0.5],
    }

    # x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0  # 3.0
    gait = torch.tensor(gaits["trotting"])
    # gait = torch.tensor(gaits["pronking"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))
    ###### -----------ldt---------------
    joint_torques = np.zeros((num_eval_steps, 12))

    obs = env.reset()

    foot_contacts = [False, False, False, False]

    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu()

        t = time.time()
        header = Header(seq=0, stamp=rospy.Time(t), frame_id="World")
        position = env.dof_pos[0, :].cpu()
        pub.publish(JointState(header, name, position, velocity, effort))
        base_pos = env.env.base_pos[0, :].cpu()
        base_quat = env.env.base_quat[0, :].cpu()
        # quat_indices = [1, 2, 3, 0]
        quat_indices = [0, 1, 2, 3]
        br.sendTransform(
            ((base_pos[0] + 5) % 10 - 5, (base_pos[1] + 5) % 10 - 5, base_pos[2]),
            (
                base_quat[quat_indices[0]],
                base_quat[quat_indices[1]],
                base_quat[quat_indices[2]],
                base_quat[quat_indices[3]],
            ),
            rospy.Time.now(),
            "Base",
            "World",
        )

        foot_data = []

        for index, foot in enumerate(env.feet_indices):
            foot_state = env.env.contact_forces[0, foot, 2] > 0
            if foot_state and not foot_contacts[index]:
                foot_data.append(env.env.foot_positions.detach().cpu().numpy())

            foot_contacts[index] = foot_state

        foot_data = np.array(foot_data)
        foot_data = foot_data.reshape(-1, 3)
        foot_data[:, :2] = (foot_data[:, :2] + 5) % 10 - 5
        sensor_pub.publish(Float32MultiArray(data=foot_data.ravel()))

        ###### -----------ldt---------------
        # joint_torques[i] = env.torques.detach().cpu().numpy()

    # plot target and measured forward velocity
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(12, 5))
    axs[0].plot(
        np.linspace(0, num_eval_steps * env.dt, num_eval_steps),
        measured_x_vels,
        color="black",
        linestyle="-",
        label="Measured",
    )
    axs[0].plot(
        np.linspace(0, num_eval_steps * env.dt, num_eval_steps),
        target_x_vels,
        color="black",
        linestyle="--",
        label="Desired",
    )
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(
        np.linspace(0, num_eval_steps * env.dt, num_eval_steps),
        joint_positions,
        linestyle="-",
        label="Measured",
    )
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    axs[2].plot(
        np.linspace(0, num_eval_steps * env.dt, num_eval_steps),
        joint_torques,
        linestyle="-",
        label="Measured",
    )
    axs[2].set_title("Joint Torques")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Joint Torques (Nm)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # to see the environment rendering, set headless=False
    play_go2(headless=False)
