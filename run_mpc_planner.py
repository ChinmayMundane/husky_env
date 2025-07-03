
import numpy as np
from mpc_planner import run_cem_planner
import os
import mujoco


inference = False
position_threshold=0.06  
rotation_threshold=0.1  
ik_pos_thresh = 2*position_threshold
ik_rot_thresh = 2*rotation_threshold
timestep = 0.05
collision_free_ik_dt = 10*timestep

#Customized parameters
results = run_cem_planner(
    # CEM parameters
    num_dof=2,
    num_batch=500,  # Use More samples for better optimization
    num_steps=10,     # Use More steps for longer planning horizon
    num_elite=0.05,   # Use More elite samples for better convergence #Int(num_elite*num_batch) is used to select elite samples
    timestep=timestep,     # Simulation Time Step Use Smaller timestep for more accurate simulation
    
    maxiter_cem=1,      # CEM iterations: Use More iterations for better convergence     
    maxiter_projection=5,   # Projection Filter iterations: Use More iterations for better Filtering
    w_pos=3.0,      # weight on position error
    w_rot=0.5,       # weight on rotation error
    w_col= 50000.0, #5000.0,      # weight on collision avoidance
    

    #Collision free IK parameters
    ik_pos_thresh=ik_pos_thresh,
    ik_rot_thresh=ik_rot_thresh,
    collision_free_ik_dt= collision_free_ik_dt,
    #Shower parameters
    show_viewer=True,
    show_contact_points=True,
    
    # Initial configuration
    initial_qpos=[0.0, 0.0], 
    
    # Target sequence
    target_names=['target_0', 'target_1'],

    #Joint limits
    max_pos= 3.0,
    max_vel= 1.0,
    max_acc= 2.0,
    max_jerk= 4.0,
    
    # Visualization
    cam_distance=15,  # View 
    
    # Convergence to target thresholds
    position_threshold=position_threshold,  # Stricter position convergence Better for more complex tasks
    rotation_threshold=rotation_threshold,   # Stricter rotation convergence Better for more complex tasks
    
    # Save Motion Related data
    save_data=False,
    data_dir=f'custom_data',
    
    # Save Point Cloud data
    cam_name="camera1",

    #Inference MLP for lamda_init and s_init
    inference=inference 
,
    #rnn = 'GRU',
    rnn = 'LSTM',

    #Stop at final target
    stop_at_final_target=False 
)


