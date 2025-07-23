# vis.py (or apply_controls.py)

import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def main():
    # --- Configuration ---
    xml_file = 'default4.xml'
    control_file = 'best_control_sequence.npy'
    planner_timestep = 0.05

    # --- Load Model and Data ---
    xml_path = os.path.join(os.path.dirname(__file__), xml_file)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file '{xml_file}' not found. Make sure it's in the same directory.")
    
    # Load the original model
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    # Create a modified XML string with the ball added
    with open(xml_path, 'r') as f:
        xml_content = f.read()
    
    # Add ball material to assets section
    if '<asset>' in xml_content:
        xml_content = xml_content.replace(
            '<asset>',
            '<asset>\n    <material name="ball_material" rgba="1 0 0 1"/>'
        )
    else:
        # If no asset section exists, add it
        xml_content = xml_content.replace(
            '<mujoco',
            '<mujoco>\n<asset>\n    <material name="ball_material" rgba="1 0 0 1"/>\n</asset>'
        )
    
    # Add ball body to worldbody section
    ball_xml = '''
    <body name="visualization_ball" pos="2.5 0 2.5">
        <geom name="ball_geom" type="sphere" size="0.3" material="ball_material"/>
        <inertial pos="0 0 0" mass="0.1" diaginertia="0.004 0.004 0.004"/>
    </body>'''
    
    if '</worldbody>' in xml_content:
        xml_content = xml_content.replace('</worldbody>', ball_xml + '\n</worldbody>')
    else:
        print("Warning: Could not find </worldbody> tag to add ball")
    
    # Create new model from modified XML
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    
    print("Added visualization ball at position (2.5, 0, 0.5)")

    # --- Load Control Sequence ---
    if not os.path.exists(control_file):
        raise FileNotFoundError(
            f"Control file '{control_file}' not found. "
            "Run 'mjx_planner.py' first to generate it."
        )
    controls = np.load(control_file)
    num_steps, num_dof = controls.shape
    print(f"Loaded control sequence from '{control_file}'")
    print(f" - Steps: {num_steps}")
    print(f" - Degrees of Freedom: {num_dof}")

    # --- Set Simulation Parameters ---
    model.opt.timestep = planner_timestep
    print(f"Set simulation timestep to {model.opt.timestep} to match planner.")

    mujoco.mj_resetData(model, data)
    
    # --- Launch Viewer and Run Simulation ---
    print("\nStarting simulation... Close the viewer window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # --- CHANGE CAMERA VIEWPOINT ---
        # Set the camera elevation (vertical angle). 
        # A more negative value means looking from higher up. -90 is top-down.
        viewer.cam.elevation = -30  # Default is often around -30. Let's make it steeper.

        # You can also adjust other camera parameters for a better view:
        # Zoom out to see more of the scene from the new angle
        viewer.cam.distance = 30
        # Rotate the camera view (0 is default)
        viewer.cam.azimuth = 90
        # Point the camera at a specific location [x, y, z]
        viewer.cam.lookat[0] = 0.25 
        # --------------------------------

        step_count = 0
        start_time = time.time()
        while viewer.is_running() and step_count < num_steps:
            sim_time = data.time
            
            data.ctrl[:num_dof] = controls[step_count]
            
            mujoco.mj_step(model, data)
            
            viewer.sync()
            
            step_count += 1

            while time.time() - start_time < data.time:
                time.sleep(0.001)

    print(f"\nSimulation finished after {step_count} steps.")


if __name__ == "__main__":
    main()