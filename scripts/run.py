#! /usr/bin/python
import isaacgym

assert isaacgym
import torch
import numpy as np
import time

import glob
import pickle as pkl

import rospy
from pathlib import Path
import rospkg
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Float32MultiArray
import tf

from go2_gym.envs import *
from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.go2.go2_config import config_go2
from go2_gym.envs.go2.velocity_tracking import VelocityTrackingEasyEnv
from isaacgym.torch_utils import quat_rotate_inverse

from tqdm import tqdm

global target_state 
target_state= np.array([0., 0., 0.])

def target_callback(data):
    global target_state
    # print("Updating target state to ", data.data)
    target_state = np.array(data.data)


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

    # # If running without ros, use the next line instead of the two subsequent ones
    # dirs = glob.glob(f"../runs/{label}/*")
    robot = Path(rospkg.RosPack().get_path("robot"))
    dirs = glob.glob(f"{robot}/go2_walking/runs/{label}/*")
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
    Cfg.terrain.num_rows = 10
    Cfg.terrain.num_cols = 10
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.mesh_type = "heightfield"
    Cfg.terrain.terrain_noise_magnitude = 0.01
    Cfg.terrain.terrain_length = 1
    Cfg.terrain.terrain_width = 1


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

    rospy.Subscriber("target_pos", Float32MultiArray, target_callback)
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
    env.set_main_agent_pose([1,1,.5],[0,0,0,1])
    

    foot_contacts = [False, False, False, False]

    for i in range(num_eval_steps):
        with torch.no_grad():
            actions = policy(obs)

        base_quat = env.env.base_quat.cpu()
        base_pos = env.env.base_pos[0].cpu().numpy()
        base_pos = base_pos - np.array([5.,5.,0.]) # TODO: Make this not hard-coded
        target = target_state[:2] - base_pos[:2]

        target_vec = torch.Tensor(np.array([[target[0], target[1], 0]]))

        target = quat_rotate_inverse(base_quat, target_vec)[0,:2].cpu().numpy()
        target_vel = target / np.sqrt(np.sum(np.square(target)))
        

        target_xvel = target_vel[0] * 1.5
        target_yvel = target_vel[1] * 0
        
        if target_vel[1] * np.sign(target_vel[0]) > 0.2:
            yaw_cmd = 1.5 * min(np.abs(target_vel[1]*3), 1)
        elif target_vel[1] * np.sign(target_vel[0]) < -0.2:
            yaw_cmd = -1.5 * min(np.abs(target_vel[1]*3), 1)
        else:
            yaw_cmd = 0


        # if np.abs(target_yvel) > 0.5:
        #     target_yvel /= a

        # yaw_cmd = 0

        # x_axis = quat_rotate_inverse(base_quat, torch.Tensor([[1.,0.,0.]]))
        # y_axis = quat_rotate_inverse(base_quat, torch.Tensor([[0.,1.,0.]]))

        base_pos = env.env.base_pos[0].cpu().numpy()
        base_pos = base_pos - np.array([5.,5.,0.]) # TODO: Make this not hard-coded
        target = target_state[:2] - base_pos[:2]
        target /= np.sqrt(np.sum(np.square(target)))
        env.commands[:, 0] = torch.Tensor([target_xvel])
        env.commands[:, 1] = torch.Tensor([target_yvel])
        env.commands[:, 2] = torch.Tensor([yaw_cmd])
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
        joint_positions[i] = env.dof_pos[0].cpu()

        t = time.time()
        header = Header(seq=0, stamp=rospy.Time(t), frame_id="World")
        position = env.dof_pos[0].cpu().numpy()
        pub.publish(JointState(header, name, position, velocity, effort))
        base_pos = env.env.base_pos[0].cpu().numpy()
        base_quat = env.env.base_quat[0].cpu().numpy()
        # quat_indices = [1, 2, 3, 0]
        quat_indices = [0, 1, 2, 3]
        br.sendTransform(
            # ((base_pos[0] + 5) % 10 - 5, (base_pos[1] + 5) % 10 - 5, base_pos[2]),
            base_pos - np.array([5,5,0]), # TODO: Make this not hard-coded
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

        if i > 100:

            for index, foot in enumerate(env.feet_indices):
                foot_state = env.env.contact_forces[0, foot, 2] > 0
                if foot_state and not foot_contacts[index]:
                    foot_data.append(env.env.foot_positions.detach().cpu().numpy())

                foot_contacts[index] = foot_state

            foot_data = np.array(foot_data)
            foot_data = foot_data.reshape(-1, 3)
            foot_data[:, :2] = (foot_data[:, :2] - 5)  # TODO: Make this not hard-coded
            sensor_pub.publish(Float32MultiArray(data=foot_data.ravel()))

        ###### -----------ldt---------------
        # joint_torques[i] = env.torques.detach().cpu().numpy()

if __name__ == "__main__":
    # to see the environment rendering, set headless=False
    play_go2(headless=False)
