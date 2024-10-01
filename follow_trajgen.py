#!/usr/bin/python3 -B

import csv
import asyncio
import math
import moteus
import time
from matplotlib import pyplot as plt
import numpy as np

# Predetermined torques from trajgen (to follow a calculated trajectory)
torque_list = []
with open('data/u_sol_traj.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        value = float(row[0])
        torque_list.append(value)

dt = 0.02 # Time step
N_traj = 500 # Number of knot points over the trajectory

# Lists to hold measured values from motor controller
torque_actual_list = []
q_actual_list_degrees = []
q_actual_list = []
qd_actual_list = []

async def main():
    c = moteus.Controller()

    # In case the controller had faulted previously, at the start of
    # this script we send the stop command in order to clear it.
    await c.set_stop()
        
    # Can control the torque via feedforward torque, advised to set kp and kd to 0 when controlling torque
    for i in range(len(torque_list)):

        time_start = time.time()
        # Send feedforward torque command to controller
        state = await c.set_position_wait_complete(position=math.nan, period_s=0.01, maximum_torque=0.2, kp_scale=0.0, kd_scale=0.0, feedforward_torque=torque_list[i], query=True)
        time_end = time.time()

        await asyncio.sleep(dt - (time_end - time_start))
        print("Execution time: ", time_end - time_start)

        # Saving actual torque, velocity and position for use in plotting
        torque_actual_list.append(state.values[moteus.Register.TORQUE])
        q_actual_list_degrees.append(state.values[moteus.Register.POSITION] * 360)
        qd_actual_list.append(state.values[moteus.Register.VELOCITY] * (2*math.pi))

        # Debug prints
        print("Iteration: ", i)
        print("Commanded Torque: ", torque_list[i])
        print("Position:", state.values[moteus.Register.POSITION])
        print("Torque: ", state.values[moteus.Register.TORQUE])
        print("dt: ", dt - (time_end - time_start) + (time_end - time_start)) # dt when accounting for time taken for set_position to occur
        print()
    
    # Recapture
    await c.set_recapture_position_velocity()
    
    # Stopping the motor at desired position for 4 sec using PD controller
    time_start = time.time()
    current_time = time_start
    while(current_time < time_start + 4):
        state = await c.set_position(position=0.5, maximum_torque=0.2, kp_scale=1.0, kd_scale=2.0, query=True)
        current_time = time.time()

    # Shutting down motor
    await c.set_stop()

    print("Final q", q_actual_list_degrees[-1]) # This was used to tune the damping coefficient, undershoot -> increase damping coeff.

    # Converting from degrees to radians
    for q in q_actual_list_degrees:
        q_actual_list.append((q / 360) * (2*math.pi))
    
    # Plotting the data
    fig, ax = plt.subplots()
    ax.set_title("Followed Trajectory: Torque vs Time")
    ax.plot(np.arange(0, N_traj-1) * dt, torque_actual_list, "-")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N/m)")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title("Followed Trajectory: Angular Position vs Time")
    ax.plot(np.arange(0, N_traj-1) * dt, q_actual_list_degrees, "-")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Position (rad)")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title("Followed Trajectory: Phase Portrait")
    ax.plot(q_actual_list, qd_actual_list, "-")
    ax.set_xlabel("Angular Position (rad)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    plt.show()
    

if __name__ == '__main__':
    asyncio.run(main())