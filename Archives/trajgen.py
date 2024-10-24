#!/usr/bin/python3 -B

import numpy as np
from matplotlib import pyplot as plt
from pydrake.all import (
    MathematicalProgram,
    Solve,
    eq,
)
import csv

# Linear or non-linear
is_linearized = False

# Physical System Constants
dt = 0.02
g = 9.81
l = 0.13
d = 0.0072
m = 0.206

# Control Constants
u_max = 0.2 # maximum force
x0 = [0.0, 0.0] # initial state
xf_desired = [np.pi - 0.2, 0] # desired final position and final velocity (note: this is the desired final state of the trajectory)

# Trajgen Constants
N_traj = 500 # Number of knot points for entire trajectory

xc = [0.0, 0.0]

def trajgen():

    def solve_for_fixed_horizon():
        prog = MathematicalProgram()

        # Linearized form
        A_arr = np.array([[0, 1], [-m*g*l*np.cos(xc[0]), -d]])
        A = np.eye(2) + dt * np.matrix(A_arr) # I + A*dt
        B = dt * np.mat("0.0; 1.0") # B*dt

        # Create decision variables
        u = prog.NewContinuousVariables(1, N_traj - 1, "u") # force input
        x = prog.NewContinuousVariables(2, N_traj, "x") # state: position and velocity

        # Add constraints
        prog.AddBoundingBoxConstraint(x0, x0, x[:, 0]) # assign initial state = x0. they wrote x0 <= x[:,0] <= x0 fsr
        for n in range(N_traj - 1):
            if (is_linearized):
                # LINEARIZED
                prog.AddConstraint(eq(x[:, n + 1], A.dot(x[:, n]) + B.dot(u[:, n])))
            else:
                # NON LINEAR
                x_1 = x[1, n]  # q'
                # x_2 = u[0, n] - d * x[1, n] - m * g * l * np.sin(x[0, n])  # q''
                x_2 = (u[0, n] - d * x[1, n] - m * g * l * np.sin(x[0, n]))/(m*l*l)  # q'' THIS IS CORRECT
                X = np.array([x_1, x_2])
                next_state = x[:, n] + dt * X
                prog.AddConstraint(eq(x[:, n + 1], next_state)) # This line is stating that the next x should be the next state

            prog.AddBoundingBoxConstraint(-u_max, u_max, u[:, n]) # input ("force") can only be between [-1, 1] 

            prog.AddQuadraticCost(u[0, n] ** 2, True) # COST (not constraint!). make sure you square the argument (it doesn't do it for you).
        prog.AddBoundingBoxConstraint(xf_desired, xf_desired, x[:, N_traj - 1])

        result = Solve(prog)
        return result, prog, x, u

    result, prog, x, u = solve_for_fixed_horizon()
    assert result.is_success(), "Optimization failed"

    u_sol_traj = result.GetSolution(u)
    x_sol_traj = result.GetSolution(x)

    fig, ax = plt.subplots()
    ax.set_title("Generated Trajectory: Phase Portait")
    ax.plot(x_sol_traj[0, :], x_sol_traj[1, :], "-")
    ax.set_xlabel("Angular Position (rad)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title("Generated Trajectory: Torque vs Time")
    ax.plot(np.arange(0, N_traj - 1) * dt, u_sol_traj.T, "-")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N/m)")
    plt.show()

    return x_sol_traj, u_sol_traj


print("Starting Trajectory Generation")
x_sol_traj, u_sol_traj = trajgen()

x_sol_traj_degrees = []
u_sol_traj_array = []
x_sol_traj_q_array = []
x_sol_traj_qd_array = []

print(x_sol_traj[0].size)
for n in range(x_sol_traj[0].size):
    x_sol_traj_degrees.append(x_sol_traj[0][n] * 180 / np.pi)
# print(x_sol_traj_degrees)

# Converting to arrays for use in other scripts
for n in range(u_sol_traj[0].size):
    u_sol_traj_array.append(u_sol_traj[0][n])

for n in range(x_sol_traj[0].size):
    x_sol_traj_q_array.append(x_sol_traj[0][n])
    x_sol_traj_qd_array.append(x_sol_traj[1][n])

print("q:\n", x_sol_traj_q_array)
print(" ")
print("qd:\n", x_sol_traj_qd_array)
print(" ")
print("u:\n", u_sol_traj_array)

# Saving values to list
with open('data/x_sol_traj_q.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for q in x_sol_traj_q_array:
        writer.writerow([q])

with open('data/x_sol_traj_qd.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for qd in x_sol_traj_qd_array:
        writer.writerow([qd])

with open('data/u_sol_traj.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for u in u_sol_traj_array:
        writer.writerow([u])


fig, ax = plt.subplots()

# Plotting the data
ax.set_title("Generated Trajectory: Angular Position vs Time")
ax.plot(np.arange(0, N_traj) * dt, x_sol_traj_degrees, "-")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angular Position (degrees)")
plt.show()