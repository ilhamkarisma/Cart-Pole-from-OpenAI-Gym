#--------------CODE IMPLEMENTATION FOR LQR----------------#
#---------Created for Final Project PSKM 2022/2023--------#
#--------------------- by Group 4 ------------------------#
import gym
import pygame
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg          # import Linalg for riccati equation

# Define the Linearized Dynamics of the System
A = np.array([[0, 1,      0, 0],
              [0, 0, -0.709, 0],
              [0, 0,      0, 1],
              [0, 0, 15.775, 0]])
B = np.array([[     0],
              [ 0.974],
              [     0],
              [-1.463]])

# Calculate the Optimal Controller
R = np.array([[0.075]])
Q = 150*np.eye(4, dtype=int)

dt = 0.02
x_array = []
i_array = []
u_array = []
tetha_array = []

# solve ricatti equation
P = linalg.solve_continuous_are(A, B, Q, R)

# calculate optimal controller gain
K = np.linalg.inv(R)@np.transpose(B)@P

# Calculate the Input Force F 
def apply_state_controller(K, x):
    u = -np.dot(K, x)   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left


#---------------- Set up the Simulation!---------------------
# Get Environment
# start time to count the computation
prev_time = pygame.time.get_ticks()
env = gym.make("CartPole-v1", render_mode="human")

# Getting Observation of the cart pole
obs = np.array(env.reset()[0])
print('\nGet initial observation of the cartpole is:\n', obs)

reward_total = 0

for i in range(1000):
    env.render()
    
    # get force direction (action) and force value (force)
    action, force = apply_state_controller(K, obs)

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -10, 10)))
    
    # change magnitute of the applied force in CartPole
    env.env.force_mag = abs_force

    # apply action
    obs, reward, done, _, _ = env.step(action)

    reward_total = reward_total+reward

    #Get Value of Pole Angle
    tetha = obs[2]

    # Calculating the needed time in system
    curr_time = pygame.time.get_ticks()
    dt = (curr_time - prev_time)/1000
    prev_time = curr_time


    if done or reward_total == 200:
        print(f'\nTerminated after {i+1} iterations.')
        print(reward_total)
        break

    x_array.append(obs)
    i_array.append(i)
    u_array.append(abs_force)
    tetha_array.append(tetha*0.0174533)

env.close()

# calculate the average of the force
print(f"Force Average = {sum(u_array)/len(u_array):.5f} Newton")
print(f"Value SSE of Theta = {float(max(tetha_array)):.5f} Radian")


#------------Plotting the system response------------------
# Mengubah Bentuk List menjadi Array 
x_array  = np.array(x_array)
i_arrray = np.array(i_array)

subplots = []
datapoints, state_num = x_array.shape

fig, ax = plt.subplots(state_num, sharex=True, sharey=True)
fig.suptitle("System Response with Step by Iterations", fontsize = 12)

for i in range(0, state_num):
  ax[i].plot(i_array[:], x_array[:, i], '-r', lw = 1.2)
  ax[i].set_axisbelow(True)
  ax[i].grid()

plt.show()