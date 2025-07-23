import numpy as np
from mjx_planner import cem_planner
import mujoco.mjx as mjx 
import mujoco
import time
import jax.numpy as jnp
import jax
import os
from mujoco import viewer
import matplotlib.pyplot as plt
from quat_math import rotation_quaternion, quaternion_multiply, quaternion_distance
import argparse
from sklearn.neighbors import NearestNeighbors
from functools import partial

import contextlib
from io import StringIO

import sys
#Enable python to search for modules in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@partial(jax.jit, static_argnames=['nvar', 'num_batch'])
def compute_xi_samples(key, xi_mean, xi_cov, nvar, num_batch):
    key, subkey = jax.random.split(key)
    xi_samples = jax.random.multivariate_normal(key, xi_mean, xi_cov+0.1*jnp.identity(nvar), (num_batch, ))
    return xi_samples, key



def run_cem_planner(
    num_dof=None,
    num_batch=None,
    num_steps=None,
    maxiter_cem=None,
    maxiter_projection=None,
    w_pos=None,
    w_rot=None,
    w_col=None,
    num_elite=None,
    timestep=None,
    initial_qpos=None,
    target_names=None,
    show_viewer=None,
    cam_distance=None,
    show_contact_points=None,
    save_data=None,
    data_dir=None,
    max_pos=None,
    max_vel=None,
    max_acc=None,
    max_jerk=None,
    cam_name=None,
    position_threshold=None,  
    rotation_threshold=None, 
    stop_at_final_target=None, 

):
    # Initialize data structures
    cost_g_list = []
    cost_r_list = []
    cost_c_list = []
    cost_list = []
    thetadot_list = []
    theta_list = []
    best_vel_list = []
    avg_primal_residual_list = []
    avg_fixed_point_residual_list = []
    best_cost_primal_residual_list = []
    best_cost_fixed_point_residual_list = []
    target_pos_list = []
    target_quat_list = []

    # Initialize CEM planner
    cem = cem_planner(
        num_dof=num_dof, 
        num_batch=num_batch, 
        num_steps=num_steps, 
        maxiter_cem=maxiter_cem,
        w_pos=w_pos,
        w_rot=w_rot,
        w_col=w_col,
        num_elite=num_elite,
        timestep=timestep,
        maxiter_projection=maxiter_projection,
        max_pos=max_pos,
        max_vel=max_vel,
        max_acc=max_acc,
        max_jerk=max_jerk
    )

    model = cem.model
    data = cem.data

    
    
    # Defining Obstacle position here is not needed as that is taken from environment
    # For mujoco (official Python bindings):
    obstacle_indices = [i for i in range(cem.model.nbody) 
                    if cem.model.body(i).name.startswith("obstacle_")]
    obst_pos = [cem.mjx_data.xpos[i] for i in obstacle_indices]
    obst_quat = [cem.mjx_data.xquat[i] for i in obstacle_indices]

    data.qpos = jnp.array(initial_qpos)
    mujoco.mj_forward(model, data)

    # Initialize point cloud generator if enabled

    # Initialize CEM variables
    xi_mean_single = jnp.zeros(cem.nvar_single)
    xi_cov_single = 10*jnp.identity(cem.nvar_single)
    xi_mean_init = jnp.tile(xi_mean_single, cem.num_dof)
    xi_cov_init = jnp.kron(jnp.eye(cem.num_dof), xi_cov_single)
    xi_mean = xi_mean_init
    xi_cov = xi_cov_init
    xi_samples, key = cem.compute_xi_samples(cem.key, xi_mean, xi_cov)
    lamda_init = jnp.zeros((cem.num_batch, cem.nvar))
    s_init = jnp.zeros((cem.num_batch, cem.num_total_constraints))

    
    timestep_counter = 0
    target_idx = 0
    current_target = target_names[target_idx]

    if show_viewer:
        with viewer.launch_passive(model, data) as viewer_:
            viewer_.cam.distance = cam_distance
            viewer_.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = show_contact_points
            
            try:
                while viewer_.is_running():
                    timestep_counter += 1
                    start_time = time.time()
        

                    # Main CEM planning loop
                    target_pos = np.array([1.5,0.0,0.0])
                    target_quat = np.array([0, 0, 0, 1])

                    if np.isnan(xi_cov).any():
                        xi_cov = xi_cov_init
                    if np.isnan(xi_mean).any():
                        xi_mean = xi_mean_init

                    try:
                        np.linalg.cholesky(xi_cov)
                    except np.linalg.LinAlgError:
                        xi_cov = xi_cov_init    

                    xi_samples, key = cem.compute_xi_samples(cem.key, xi_mean, xi_cov)
                    xi_samples_reshaped = xi_samples.reshape(cem.num_batch, cem.num_dof, cem.nvar_single)
                        

                    # CEM computation
                    cost, best_ctrl, best_traj, \
                    xi_mean, xi_cov, ctrl, base_pos, avg_primal_res, avg_fixed_res, \
                    primal_res, fixed_res, idx_min = cem.compute_cem(
                        xi_mean,
                        xi_cov,
                        data.qpos[:num_dof],
                        data.ctrl[:num_dof],
                        data.qacc[:num_dof],
                        target_pos,
                        target_quat,
                        lamda_init,
                        s_init,
                        xi_samples
                    )

                    # --- CHECK TARGET CONVERGENCE ---
                    # Calculate distance to the target in the XY plane
                    robot_pos_xy = data.xpos[cem.base_id][:2]
                    target_pos_xy = target_pos[:2]
                    distance_to_target = np.linalg.norm(robot_pos_xy - target_pos_xy)

                    # Check if the robot has reached the current target
                    if distance_to_target < position_threshold:
                        print(f"\n--- Target at {target_pos} reached! (Distance: {distance_to_target:.3f}m) ---")
                        
                        # In this script, there's only one target, which is the final one.
                        # If we should stop at the final target, break the loop.
                        if stop_at_final_target:
                            print("--- Final target reached. Stopping simulation. ---")
                            break # This will exit the `while viewer_.is_running()` loop.

                    # Check target convergence
                    # current_cost_g = np.linalg.norm(data.site_xpos[cem.tcp_id] - target_pos)   
                    # current_cost_r = quaternion_distance(data.xquat[cem.hande_id], target_quat)
                    current_cost = np.round(cost, 2)
                    
                    # if current_cost_g < position_threshold and current_cost_r < rotation_threshold:
                    #     if target_idx == len(target_names) - 1:
                    #         if stop_at_final_target:
                    #             data.qvel[:num_dof] = np.zeros(num_dof)
                    #         else:
                    #             target_idx = 0
                    #     else:
                    #         target_idx += 1
                    #     current_target = target_names[target_idx]
                    
                    # #ACtivate  collision free IK if cost position/rotation is less than 2*threshold
                    # if current_cost_g < ik_pos_thresh and current_cost_r < ik_rot_thresh:
                    #     collision_free_ik = True
                    # else:
                    #     collision_free_ik = False


                    # Apply control as per MPC coupled with  CEM
                    vel_action = np.mean(best_ctrl[1:int(num_steps*0.9)], axis=0)
                    # print(f"vel_action: {vel_action}")

                    data.ctrl[:num_dof] = vel_action
                    mujoco.mj_step(model, data)
                    # print(f"data.qpos: {data.qpos}")
                    # print(f"data.qvel: {data.qvel}")   

                    # Store data
      
                    thetadot_list.append(vel_action)
                    theta_list.append(data.qpos[:num_dof].copy())
                    
                    cost_list.append(current_cost[-1] if isinstance(current_cost, np.ndarray) else current_cost)


                    # # cost_list.append(float(cost[-1]) if isinstance(cost, np.ndarray) else float(cost))
                    # if isinstance(cost, np.ndarray):
                    #     cost_list.append(float(np.asarray(cost).squeeze()[-1]))
                    # else:
                    #     cost_list.append(float(cost))

                    best_vel_list.append(best_ctrl)
                    
                    avg_primal_residual_list.append(np.mean(avg_primal_res, axis=1))
                    avg_fixed_point_residual_list.append(np.mean(avg_fixed_res, axis=1))
                    best_cost_primal_residual_list.append(avg_primal_res[:, idx_min])
                    best_cost_fixed_point_residual_list.append(avg_fixed_res[:, idx_min])
                    
                    if not any(np.allclose(pos_, target_pos) for pos_ in target_pos_list):
                        target_pos_list.append(target_pos)
                    if not any(np.allclose(quat_, target_quat) for quat_ in target_quat_list):
                        target_quat_list.append(target_quat)
                    
                    
                    # Print status

                    print(f'Step Time: {"%.0f"%((time.time() - start_time)*1000)}ms  | Cost: {current_cost}')
  

                    # Update viewer
                    viewer_.sync()
                    time_until_next_step = model.opt.timestep - (time.time() - start_time)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

            except KeyboardInterrupt:
                print("Interrupted by user!")
            
            finally:
                
                # Save Motion data
                if save_data:
                    print("Saving Motion, Target and Obstacle data ...")
                    os.makedirs(data_dir, exist_ok=True)

                    #Saving Motion data
                    print("Saving Motion data...")
                    np.savetxt(f'{data_dir}/costs.csv', cost_list, delimiter=",")
                    np.savetxt(f'{data_dir}/thetadot.csv', thetadot_list, delimiter=",")
                    np.savetxt(f'{data_dir}/theta.csv', theta_list, delimiter=",")
                    np.savetxt(f'{data_dir}/cost_g.csv', cost_g_list, delimiter=",")
                    np.savetxt(f'{data_dir}/cost_r.csv', cost_r_list, delimiter=",")
                    np.savetxt(f'{data_dir}/cost_c.csv', cost_c_list, delimiter=",")
                    np.savetxt(f'{data_dir}/avg_primal_residual.csv', avg_primal_residual_list, delimiter=",")
                    np.savetxt(f'{data_dir}/avg_fixed_point_residual.csv', avg_fixed_point_residual_list, delimiter=",")
                    np.savetxt(f'{data_dir}/best_cost_primal_residual.csv', best_cost_primal_residual_list, delimiter=",")
                    np.savetxt(f'{data_dir}/best_cost_fixed_point_residual.csv', best_cost_fixed_point_residual_list, delimiter=",")
                    np.save(f'{data_dir}/best_vels.npy', np.array(best_vel_list))
                    print("Motion data saved!")
                    
                    # Save Target positions and orientations
                    print("Saving Target positions and orientations...")
                    np.savetxt(f'{data_dir}/target_positions.csv', target_pos_list, delimiter=",")
                    np.savetxt(f'{data_dir}/target_quaternions.csv', target_quat_list, delimiter=",")
                    print("Target positions and orientations saved!")
                    # Save Obstacle positions and orientations
                    print("Saving Obstacle positions and orientations...")
                    np.savetxt(f'{data_dir}/obstacle_positions.csv', obst_pos, delimiter=",")
                    np.savetxt(f'{data_dir}/obstacle_quaternions.csv', obst_quat, delimiter=",")
                    print("Obstacle positions and orientations saved!")
                    print(f"Motion, Target and Obstacle data saved to {data_dir}")



    return {
        'cost_g': cost_g_list,
        'cost_r': cost_r_list,
        'cost_c': cost_c_list,
        'cost': cost_list,
        'thetadot': thetadot_list,
        'theta': theta_list,
        'best_vels': best_vel_list,
        'primal_residual': avg_primal_residual_list,
        'fixed_point_residual': avg_fixed_point_residual_list,
        'best_cost_primal_residual': best_cost_primal_residual_list,
        'best_cost_fixed_point_residual': best_cost_fixed_point_residual_list
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CEM planner with configurable parameters')
    
    # CEM planner parameters
    parser.add_argument('--num_dof', type=int, default=6)
    parser.add_argument('--num_batch', type=int, default=1000)
    parser.add_argument('--num_steps', type=int, default=16)
    parser.add_argument('--maxiter_cem', type=int, default=1)
    parser.add_argument('--w_pos', type=float, default=20.0)
    parser.add_argument('--w_rot', type=float, default=3.0)
    parser.add_argument('--w_col', type=float, default=10.0)
    parser.add_argument('--num_elite', type=float, default=0.05)
    parser.add_argument('--timestep', type=float, default=0.05)
    parser.add_argument('--maxiter_projection', type=int, default=5)
    
    # Initial configuration
    parser.add_argument('--initial_qpos', type=float, nargs='+', default=[0.0, 0.0])
    
    # Visualization options
    parser.add_argument('--no_viewer', action='store_true')
    parser.add_argument('--cam_distance', type=float, default=4)
    parser.add_argument('--no_contact_points', action='store_true')
    parser.add_argument('--position_threshold', type=float, default=0.1, help="Distance in meters to consider a target reached.")
    parser.add_argument('--rotation_threshold', type=float, default=0.3) # Kept for consistency, though unused in this version
    parser.add_argument('--continue_after_final', action='store_true', help="If set, the planner will not stop at the final target.")
    
    # Convergence criteria
    parser.add_argument('--position_threshold', type=float, default=0.04)
    parser.add_argument('--rotation_threshold', type=float, default=0.3)
    
    # Target sequence
    parser.add_argument('--targets', type=str, nargs='+', default=["target_0", "target_1", "home"])
    
    # Save data
    parser.add_argument('--save_data', action='store_true')
    parser.add_argument('--data_dir', type=str, default='data')
    
    # MLP parameters
    parser.add_argument('--max_pos', type=float, default=None)
    parser.add_argument('--max_vel', type=float, default=None)
    parser.add_argument('--max_acc', type=float, default=None)
    parser.add_argument('--max_jerk', type=float, default=None)
    
    # Point cloud parameters

    parser.add_argument('--cam_name', type=str, default="camera1")
    
    args = parser.parse_args()
    
    run_cem_planner(
        num_dof=args.num_dof,
        num_batch=args.num_batch,
        num_steps=args.num_steps,
        maxiter_cem=args.maxiter_cem,
        maxiter_projection=args.maxiter_projection,
        w_pos=args.w_pos,
        w_rot=args.w_rot,
        w_col=args.w_col,
        num_elite=args.num_elite,
        timestep=args.timestep,
        initial_qpos=args.initial_qpos,
        target_names=args.targets,
        show_viewer=not args.no_viewer,
        cam_distance=args.cam_distance,
        show_contact_points=not args.no_contact_points,
        position_threshold=args.position_threshold,
        rotation_threshold=args.rotation_threshold,
        save_data=args.save_data,
        data_dir=args.data_dir,
        stop_at_final_target=not args.continue_after_final,
        inference=args.inference,
        rnn=args.rnn,
        max_pos=args.max_pos,
        max_vel=args.max_vel,
        max_acc=args.max_acc,
        max_jerk=args.max_jerk,
        # position_threshold=args.position_threshold,
        # rotation_threshold=args.rotation_threshold,
        # stop_at_final_target=not args.continue_after_final,

        cam_name=args.cam_name
    )