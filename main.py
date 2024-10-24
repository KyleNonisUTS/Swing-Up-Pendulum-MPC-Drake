#!/usr/bin/python3 -B

import numpy as np
from matplotlib import pyplot as plt
from pydrake.all import (
    MathematicalProgram,
    Solve,
    eq,
)
import csv
import asyncio
import math
import moteus
import time

#################################   BEGIN USER VARIABLES   #################################

# Physical System Constants
dt = 0.02
g = 9.81
l = 0.13
d = 0.0072
m = 0.206

# Control Constants
u_max = 0.2 # maximum torque
x0 = [0.0, 0.0] # initial state
xf_desired = [np.pi - 0.2, 0] # desired final position and final velocity (note: this is the desired final state of the trajectory)

# Trajgen Constants
N_traj = 500 # Number of knot points for entire trajectory

# MPC Constants
N_mpc = 15 # Horizon knot points
threshold = 0.1 # Error tolerance between final and current state

xc = [0.0, 0.0]

verbose = False # more print outs for debugging

#################################   END USER VARIABLES   #################################

def solve_for_trajectory():
    # Create mathematical program object
    prog = MathematicalProgram()

    # Create decision variables
    u = prog.NewContinuousVariables(1, N_traj - 1, "u") # force input
    x = prog.NewContinuousVariables(2, N_traj, "x") # state: position and velocity

    # Add constraints
    prog.AddBoundingBoxConstraint(x0, x0, x[:, 0]) # assign initial state = x0
    for n in range(N_traj - 1):
        # NON LINEAR
        x_1 = x[1, n]  # q'
        x_2 = (u[0, n] - d * x[1, n] - m * g * l * np.sin(x[0, n]))/(m*l*l)  # q''
        X = np.array([x_1, x_2])
        next_state = x[:, n] + dt * X
        prog.AddConstraint(eq(x[:, n + 1], next_state)) # This line is stating that the next x should be the next state

        prog.AddBoundingBoxConstraint(-u_max, u_max, u[:, n]) # input ("force") can only be between [-1, 1] 

        prog.AddQuadraticCost(u[0, n] ** 2, True) # COST (not constraint!). make sure you square the argument (it doesn't do it for you).
    prog.AddBoundingBoxConstraint(xf_desired, xf_desired, x[:, N_traj - 1])

    result = Solve(prog)
    return result, prog, x, u

def solve_for_fixed_horizon(xc, uc, xf, N_mpc):
    if verbose: print("uc:", uc, " xc:", xc, " xf:", xf, " N:", N_mpc)
    prog = MathematicalProgram()

    # Create decision variables
    u = prog.NewContinuousVariables(1, N_mpc - 1, "u") # force input
    x = prog.NewContinuousVariables(2, N_mpc, "x") # state: position and velocity

    # Add constraints
    prog.AddBoundingBoxConstraint(xc, xc, x[:, 0]) # assign initial state
    for n in range(N_mpc - 1):
        # NON-LINEAR
        x_1 = x[1, n]  # q'
        x_2 = (u[0, n] - d * x[1, n] - m * g * l * np.sin(x[0, n]))/(m*l*l)  # q'' THIS IS CORRECT
        X = np.array([x_1, x_2])
        next_state = x[:, n] + dt * X
        prog.AddConstraint(eq(x[:, n + 1], next_state)) # This line is stating that the next x should be the next state

        prog.AddBoundingBoxConstraint(-u_max, u_max, u[:, n]) # input ("force") limits

        prog.AddQuadraticCost(u[0, n] ** 2, True) # COST (not constraint!). make sure you square the argument (it doesn't do it for you).
    prog.AddBoundingBoxConstraint(xf, xf, x[:, N_mpc - 1])

    result = Solve(prog)
    return result, prog, x, u

def plotter(title, xlabel, ylabel, xdata, ydata):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(xdata, ydata, "-")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def write_to_csv(path, data):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for d in data:
            writer.writerow([d])

def trajgen():
    x_sol_traj_q_list = []
    x_sol_traj_qd_list = []
    u_sol_traj_list = []

    print("Starting Trajectory Generation... (This may take a while)")
    result, prog, x, u = solve_for_trajectory()
    assert result.is_success(), "Trajectory Generation: Optimization failed"
    print("Trajectory Generation: SUCCESS")

    u_sol_traj = result.GetSolution(u)
    x_sol_traj = result.GetSolution(x)

    plotter("Simulated Trajectory: Phase Portait", "Angular Position (rad)", "Angular Velocity (rad/s)", x_sol_traj[0, :], x_sol_traj[1, :])
    plotter("Simulated Trajectory: Torque vs Time", "Time (s)", "Torque (N/m)", np.arange(0, N_traj - 1) * dt, u_sol_traj.T)
    plotter("Simulated Trajectory: Angular Position vs Time", "Time (s)", "Angular Position (rad)", np.arange(0, N_traj) * dt, x_sol_traj[0, :])

    if verbose: print(x_sol_traj[0].size)

    # Converting to arrays for use in other scripts
    for n in range(u_sol_traj[0].size):
        u_sol_traj_list.append(u_sol_traj[0][n])

    for n in range(x_sol_traj[0].size):
        x_sol_traj_q_list.append(x_sol_traj[0][n])
        x_sol_traj_qd_list.append(x_sol_traj[1][n])

    if verbose: 
        print("q:\n", x_sol_traj_q_list)
        print(" ")
        print("qd:\n", x_sol_traj_qd_list)
        print(" ")
        print("u:\n", u_sol_traj_list)

    # Saving to csv
    write_to_csv('data/trajgen/x_sol_traj_q.csv', x_sol_traj_q_list)
    write_to_csv('data/trajgen/x_sol_traj_qd.csv', x_sol_traj_qd_list)
    write_to_csv('data/trajgen/u_sol_traj.csv', u_sol_traj_list)

    print("Generated trajectory saved to csv files in data folder")

async def pd():
    i = 0
    q_actual_list = []
    qd_actual_list = []
    torque_actual_list = []

    print("Starting exclusive pd control")

    c = moteus.Controller()

    # In case the controller had faulted previously, at the start of
    # this script we send the stop command in order to clear it.
    await c.set_stop()

    # Stopping the motor at desired position for 10 sec using PD controller
    time_start = time.time()
    current_time = time_start
    while(current_time < time_start + N_traj * dt):
        time_start_command = time.time()
        state = await c.set_position(position=0.5, maximum_torque=u_max, kp_scale=1.0, kd_scale=1.0, query=True)
        torque_actual_list.append(state.values[moteus.Register.TORQUE])
        q_actual_list.append(state.values[moteus.Register.POSITION] * (2*math.pi))
        qd_actual_list.append(state.values[moteus.Register.VELOCITY] * (2*math.pi))
        time_end_command = time.time()
        await asyncio.sleep(dt - (time_end_command - time_start_command))
        current_time = time.time()
        i += 1
    
    # Shutting down motor
    await c.set_stop()

    plotter("Exclusive PD Control: Torque vs Time", "Time (s)", "Torque (N/m)", np.arange(0, i) * dt, torque_actual_list)
    plotter("Exclusive PD Control: Angular Position vs Time", "Time (s)", "Angular Position (rad)", np.arange(0, i) * dt, q_actual_list)
    plotter("Exclusive PD Control: Phase Portrait", "Angular Position (rad)", "Angular Velocity (rad/s)", q_actual_list, qd_actual_list)

    # Saving to csv
    write_to_csv('data/pd/q_actual.csv', q_actual_list)
    write_to_csv('data/pd/qd_actual.csv', qd_actual_list)
    write_to_csv('data/pd/u_actual.csv', torque_actual_list)

async def feedforward_control():
    q_actual_list = []
    qd_actual_list = []
    torque_actual_list = []

    # Predetermined torques from trajgen (to follow a calculated trajectory)
    torque_list = []
    with open('data/trajgen/u_sol_traj.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            value = float(row[0])
            torque_list.append(value)
    
    print("Starting feedforward control...")
    c = moteus.Controller()

    # In case the controller had faulted previously, at the start of
    # this script we send the stop command in order to clear it.
    await c.set_stop()
        
    # Can control the torque via feedforward torque, advised to set kp and kd to 0 when controlling torque
    for i in range(len(torque_list)):

        time_start = time.time()
        # Send feedforward torque command to controller
        state = await c.set_position_wait_complete(position=math.nan, period_s=0.01, maximum_torque=u_max, kp_scale=0.0, kd_scale=0.0, feedforward_torque=torque_list[i], query=True)
        time_end = time.time()

        await asyncio.sleep(dt - (time_end - time_start))
        if verbose: print("Execution time: ", time_end - time_start)

        # Saving actual torque, velocity and position for use in plotting
        torque_actual_list.append(state.values[moteus.Register.TORQUE])
        q_actual_list.append(state.values[moteus.Register.POSITION] * (2*math.pi))
        qd_actual_list.append(state.values[moteus.Register.VELOCITY] * (2*math.pi))

        # Debug prints
        if verbose:
            print("Iteration: ", i)
            print("Commanded Torque: ", torque_list[i])
            print("Position:", state.values[moteus.Register.POSITION])
            print("Torque: ", state.values[moteus.Register.TORQUE])
            print("dt: ", dt - (time_end - time_start) + (time_end - time_start)) # dt when accounting for time taken for set_position to occur
            print()
    
    print("Feedforward trajectory complete, starting PD controller to hold upright")
    # Recapture
    await c.set_recapture_position_velocity()

    # Stopping the motor at desired position for 4 sec using PD controller
    time_start = time.time()
    current_time = time_start
    while(current_time < time_start + 4):
        state = await c.set_position(position=0.5, maximum_torque=u_max, kp_scale=1.0, kd_scale=1.0, query=True)
        current_time = time.time()
    
    # Shutting down motor
    await c.set_stop()

    if verbose: print("Final q", q_actual_list[-1]) # This was used to tune the damping coefficient, undershoot -> increase damping coeff.

    plotter("Feedforward Control: Torque vs Time", "Time (s)", "Torque (N/m)", np.arange(0, N_traj - 1) * dt, torque_actual_list)
    plotter("Feedforward Control: Angular Position vs Time", "Time (s)", "Angular Position (rad)", np.arange(0, N_traj - 1) * dt, q_actual_list)
    plotter("Followed Trajectory: Phase Portrait", "Angular Position (rad)", "Angular Velocity (rad/s)", q_actual_list, qd_actual_list)

    # Saving to csv
    write_to_csv('data/feedforward/q_actual.csv', q_actual_list)
    write_to_csv('data/feedforward/qd_actual.csv', qd_actual_list)
    write_to_csv('data/feedforward/u_actual.csv', torque_actual_list)

async def mpc():
    q_actual_list = []
    qd_actual_list = []
    torque_actual_list = []

    global N_mpc
    xc = x0 # current state
    uc = 0.0 # current input

    x_sol_traj_q = []

    with open('data/trajgen/x_sol_traj_q.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            value = float(row[0])
            x_sol_traj_q.append(value)

    # Trajectory generated from trajgen.py: Angular Velocity
    x_sol_traj_qd = []
    with open('data/trajgen/x_sol_traj_qd.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            value = float(row[0])
            x_sol_traj_qd.append(value)

    x_sol_traj = [x_sol_traj_q, x_sol_traj_qd]

    c = moteus.Controller()

    # In case the controller had faulted previously, at the start of
    # this script we send the stop command in order to clear it.
    await c.set_stop()
        
    # MPC loop
    # This loop continues until the current state is close enough to the final state.
    print("Starting model predictive control loop")
    i = 0
    xf = [x_sol_traj[0][N_mpc - 1], x_sol_traj[1][N_mpc - 1]] # First solution is N_mpc - 1 steps ahead, from trajopt
    while(abs(xc[0] - xf_desired[0]) > threshold or abs(xc[1] - xf_desired[1]) > threshold):
        if verbose:
            print(" ")
            print("Loop iteration: ", i)
        time_start = time.time()
        # x_sol_traj_segment = x_sol_traj[0][i : N_mpc + i]
        result, prog, x, u = solve_for_fixed_horizon(xc, uc, xf, N_mpc)
        # Checking if loop iteration is not near the end of the trajectory
        if (i < len(x_sol_traj[0]) - N_mpc):
            xf = [x_sol_traj[0][N_mpc + i], x_sol_traj[1][N_mpc + i]] # First solution is N_mpc steps ahead, from trajopt RIGHT!        # if near the end, use final state and reduce mpc horizon
        else:
            xf = xf_desired
            N_mpc -= 1
            # if MPC horizon is too small, break from loop
            if (N_mpc < 2):
                break
        success = result.is_success()
        if not success:
            if verbose: print("OPTIMIZATION FAILED") # This generally isnt the end of the world, as the PD controller can take over
            break
        u_sol = result.GetSolution(u)
        x_sol = result.GetSolution(x)

        # Set first torque from MP solution as the current torque
        uc = u_sol[0][0] # Applied force
        if verbose: print("Commanded Torque: ", uc)

        # Applying the torque, allowing 0.01s for command to complete
        state = await c.set_position_wait_complete(position=math.nan, maximum_torque=u_max, period_s=0.01, kp_scale=0.0, kd_scale=0.0, feedforward_torque=uc, query=True)
        time_end = time.time()

        await asyncio.sleep(dt - (time_end - time_start)) # dt - computation and command time
        xc = [state.values[moteus.Register.POSITION] * (2*3.14), state.values[moteus.Register.VELOCITY] * (2*3.14)] # Resulting state (Had to convert from revoultions to radians)
        uc = state.values[moteus.Register.TORQUE] # Actual torque applied by controller

        # # Saving actual torque and position for use in plotting
        torque_actual_list.append(state.values[moteus.Register.TORQUE])
        q_actual_list.append(state.values[moteus.Register.POSITION] * (2*math.pi))
        qd_actual_list.append(state.values[moteus.Register.VELOCITY] * (2*math.pi))

        # Debug prints
        if verbose:
            print("Controller Torque: ", state.values[moteus.Register.TORQUE])
            print("Execution Time: ", time_end - time_start)

        # Iterate MPC loop
        i += 1
    
    # Recapture
    await c.set_recapture_position_velocity()

    # Stopping the motor at desired position for 4 sec using PD controller
    time_start = time.time()
    current_time = time_start
    while(current_time < time_start + 4):
        state = await c.set_position(position=0.5, maximum_torque=u_max, kp_scale=1.0, kd_scale=1.0, query=True)
        current_time = time.time()

    # Shutting down motor
    await c.set_stop()

    plotter("MPC: Measured Torque vs Time", "Time (s)", "Measured Torque (N/m)", np.arange(0, i) * dt, torque_actual_list)
    plotter("MPC: Measured Angular Position vs Time", "Time (s)", "Measured Angular Position (rad)", np.arange(0, i) * dt, q_actual_list)
    plotter("MPC: Phase Portrait", "Measured Angular Position (rad)", "Measured Angular Velocity (rad/s)", q_actual_list, qd_actual_list)

    # Saving to csv
    write_to_csv('data/mpc/q_actual.csv', q_actual_list)
    write_to_csv('data/mpc/qd_actual.csv', qd_actual_list)
    write_to_csv('data/mpc/u_actual.csv', torque_actual_list)

if __name__ == '__main__':
        print("========================================================================================")
        print("Select option: 'trajgen', 'pd', 'feedforward', 'mpc', 'quit' (case sensitive)")
        print("NOTE: trajgen must be run first to generate trajectories for the other functions to use (except pd)")
        print("========================================================================================")
        print(" ")

        user_input = input("OPTION: ")
        print(" ")

        match user_input:
            case "trajgen":
                trajgen()
            case "pd":
                asyncio.run(pd())
            case "feedforward":
                asyncio.run(feedforward_control())
            case "mpc":  
                asyncio.run(mpc())
            case "quit":
                pass
            case _:
                print("unknown request")