
import numpy as np
from mpc_planner import run_cem_planner
import os
import mujoco


timestep = 0.05

#Customized parameters
results = run_cem_planner(
    # CEM parameters
    num_dof=2,
    num_batch=10,  # Use More samples for better optimization
    num_steps=2,     # Use More steps for longer planning horizon
    num_elite=0.5,   # Use More elite samples for better convergence #Int(num_elite*num_batch) is used to select elite samples
    timestep=timestep,     # Simulation Time Step Use Smaller timestep for more accurate simulation
    
    maxiter_cem=1,      # CEM iterations: Use More iterations for better convergence     
    maxiter_projection=2,   # Projection Filter iterations: Use More iterations for better Filtering
    w_pos=3.0,      # weight on position error
    w_rot=0.5,       # weight on rotation error
    w_col= 50.0, #5000.0,      # weight on collision avoidance
    

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

    
    # Save Motion Related data
    save_data=False,
    data_dir=f'custom_data',
    
    # Save Point Cloud data
    cam_name="camera1",

)


