#!/usr/bin/python3 -B

import csv
import numpy as np
from matplotlib import pyplot as plt
from pydrake.all import (
    MathematicalProgram,
    Solve,
    eq,
)
import asyncio
import math
import moteus
import time

# Linear or non-linear
is_linearized = False

# Control Constants
u_max = 0.2 # maximum force
x0 = [0.0, 0.0] # initial state
xf_desired = [np.pi - 0.2, 0] # desired final position and final velocity (note: this is the desired final state of the trajectory)

# Physical System Constants
dt = 0.02
g = 9.81
l = 0.13
d = 0.0072
m = 0.206

# MPC Constants
threshold = 0.1 # Tolerance between final and current state

# Trajectory generated from trajgen.py: Angular Position
x_sol_traj_q = []
with open('data/x_sol_traj_q.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        value = float(row[0])
        x_sol_traj_q.append(value)

# Trajectory generated from trajgen.py: Angular Velocity
x_sol_traj_qd = []
with open('data/x_sol_traj_qd.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        value = float(row[0])
        x_sol_traj_qd.append(value)
x_sol_traj = [x_sol_traj_q, x_sol_traj_qd]

x0 = [0.0, 0.0] # initial state

# Lists to hold measured values from motor controller
torque_actual_list = []
q_actual_list = []
qd_actual_list = []

def solve_for_fixed_horizon(xc, uc, x_sol_traj_segment, xf, N_mpc):
        # print("uc:", uc, " xc:", xc, " x_sol_traj_segment:", x_sol_traj_segment, " xf:", xf, " N:", N_mpc)
        print("uc:", uc, " xc:", xc, " xf:", xf, " N:", N_mpc)
        prog = MathematicalProgram()

        # Create decision variables
        u = prog.NewContinuousVariables(1, N_mpc - 1, "u") # force input
        x = prog.NewContinuousVariables(2, N_mpc, "x") # state: position and velocity

        # Linearized form
        # A_arr = np.array([[0, 1], [-m_actual * g * l * np.cos(xc[0]), -d]])
        # print(A_arr)
        # A = np.eye(2) + dt * np.matrix(A_arr) # I + A*dt
        # B = dt * np.mat("0.0; 1.0") # B*dt

        # Add constraints
        prog.AddBoundingBoxConstraint(xc, xc, x[:, 0]) # assign initial state
        for n in range(N_mpc - 1):
            if (is_linearized):
                # LINEAR
                A_arr = np.array([[0, 1], [-m * g * l * np.cos(x_sol_traj_segment[n]), -d]])
                A = np.eye(2) + dt * np.matrix(A_arr) # I + A*dt
                B = dt * np.mat("0.0; 1.0") # B*dt
                prog.AddConstraint(eq(x[:, n + 1], A.dot(x[:, n]) + B.dot(u[:, n])))
            else:
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

async def main():
    N_mpc = 15 # Horizon knot points
    xc = x0 # current state
    uc = 0.0 # current input

    c = moteus.Controller()

    # In case the controller had faulted previously, at the start of
    # this script we send the stop command in order to clear it.
    await c.set_stop()
        
    # Main control loop (MPC?)
    # This loop continues until the current state is close enough to the final state.
    i = 0
    xf = [x_sol_traj[0][N_mpc - 1], x_sol_traj[1][N_mpc - 1]] # First solution is N_mpc - 1 steps ahead, from trajopt
    while(abs(xc[0] - xf_desired[0]) > threshold or abs(xc[1] - xf_desired[1]) > threshold):
    # while(i < 400):
        print(" ")
        print("Iteration: ", i)
        time_start = time.time()
        x_sol_traj_segment = x_sol_traj[0][i : N_mpc + i]
        result, prog, x, u = solve_for_fixed_horizon(xc, uc, x_sol_traj_segment, xf, N_mpc)
        # Checking if loop iteration is not near the end of the trajectory
        if (i < len(x_sol_traj[0]) - N_mpc):
            xf = [x_sol_traj[0][N_mpc + i], x_sol_traj[1][N_mpc + i]] # First solution is N_mpc steps ahead, from trajopt RIGHT!        # if near the end, use final state and reduce mpc horizon
        else:
            xf = xf_desired
            N_mpc -= 1
            # if MPC horizon is too small, break from loop
            if (N_mpc < 2):
                break
        # assert result.is_success(), "Optimization failed"
        success = result.is_success()
        if not success:
            print("OPTIMIZATION FAILED")
            # await c.set_stop()
            break
        u_sol = result.GetSolution(u)
        x_sol = result.GetSolution(x)

        # Set first torque from MP solution as the current torque
        uc = u_sol[0][0] # Applied force
        print("Commanded Torque: ", uc)

        # Applying the torque, allowing 0.01s for command to complete
        state = await c.set_position_wait_complete(position=math.nan, maximum_torque=0.2, period_s=0.01, kp_scale=0.0, kd_scale=0.0, feedforward_torque=uc, query=True)
        time_end = time.time()

        await asyncio.sleep(dt - (time_end - time_start)) # dt - computation and command time
        xc = [state.values[moteus.Register.POSITION] * (2*3.14), state.values[moteus.Register.VELOCITY] * (2*3.14)] # Resulting state (Had to convert from revoultions to radians)
        uc = state.values[moteus.Register.TORQUE] # Actual torque applied by controller

        # # Saving actual torque and position for use in plotting
        torque_actual_list.append(state.values[moteus.Register.TORQUE])
        q_actual_list.append(state.values[moteus.Register.POSITION])
        qd_actual_list.append(state.values[moteus.Register.VELOCITY])

        # Debug prints
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
        state = await c.set_position(position=0.5, maximum_torque=0.2, kp_scale=1.0, kd_scale=1.0, query=True)
        current_time = time.time()

    # Shutting down motor
    await c.set_stop()

    # Plotting data
    fig, ax = plt.subplots()
    ax.set_title("MPC: Measured Torque vs Time")
    ax.plot(np.arange(0, i) * dt, torque_actual_list, "-")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Measured Torque (N/m)")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title("MPC: Measured Angular Position vs Time")
    ax.plot(np.arange(0, i) * dt, q_actual_list, "-")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Measured Angular Position (rad)")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title("MPC: Phase Portrait")
    ax.plot(q_actual_list, qd_actual_list, "-")
    ax.set_xlabel("Measured Angular Position (rad)")
    ax.set_ylabel("Measured Angular Velocity (rad)")
    plt.show()

if __name__ == '__main__':
    asyncio.run(main())