
import numpy as np
from mpc_planner import run_cem_planner
import os
import mujoco



timestep = 0.05

#Customized parameters
results = run_cem_planner(
    # CEM parameters
    num_dof=2,
    num_batch=50,  # Use More samples for better optimization
    num_steps=100,     # Use More steps for longer planning horizon
    num_elite=0.2,   # Use More elite samples for better convergence #Int(num_elite*num_batch) is used to select elite samples
    timestep=timestep,     # Simulation Time Step Use Smaller timestep for more accurate simulation
    
    maxiter_cem=1,      # CEM iterations: Use More iterations for better convergence     
    maxiter_projection=5,   # Projection Filter iterations: Use More iterations for better Filtering
    w_pos=1.0,      # weight on position error
    w_rot=0.5,       # weight on rotation error
    w_col= 5.0, #5000.0,      # weight on collision avoidance
    

    #Shower parameters
    show_viewer=True,
    show_contact_points=True,
    
    position_threshold=1.0,  # Stop when robot is within 1 of the target
    stop_at_final_target=True, # Tell the script to actually stop

    # Initial configuration
    # initial_qpos=[0.0, 0.0], 
    initial_qpos = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    
    # Target sequence
    target_names=['target_0', 'target_1'],

    #Joint limits
    max_pos= 5.0,
    max_vel= 100.0,
    max_acc= 600.0,
    max_jerk= 60.0,
    
    # Visualization
    cam_distance=15,  # View 
  
    # Save Motion Related data
    save_data=False,
    data_dir=f'custom_data',
    
    # Save Point Cloud data
    cam_name="camera1",

)


