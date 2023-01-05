#----------CODE IMPLEMENTATION FOR Compensator-------------#
#---------Created for Final Project PSKM 2022/2023--------#
#-------------------- by Group 4 ----------------------#
import gym
import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib import style

# Define the Linearized Dynamics of the System
A = np.array([[0, 1,      0, 0],
              [0, 0, -0.709, 0],
              [0, 0,      0, 1],
              [0, 0, 15.775, 0]])
B = np.array([[     0],
              [ 0.974],
              [     0],
              [-1.466]])
C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
D = np.array([[0],
              [0]])

#Desired Pole Placement K
poleK = [-1, -5, -10, -15]
print('Desired Pole Placement for K is :' , poleK)
#Desired Pole Placement L
poleL = [-1, -3, -5,  -10]
print('Desired Pole Placement for L is :' , poleL)

#Input Matrix K and L
K = np.array([[-52.3544, -71.5510, -253.5935,  -68.6839]])

L = np.array([[13.2701,  -1.9944],
              [36.0421, -11.6834],
              [-1.9286,   8.7299],
              [-10.4727, 26.7325]])

x_hat = np.array([[0],
                  [0],
                  [0],
                  [0]])

dt = 0.001
x_array = []
i_array = []
x_hat_array = []
t_array = []
u_array = []
tetha_array = []

# Calculate the Input Force F 
def apply_state_controller(K, x):
    # State Feedback
    u = -K@x
        
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left

# Calculate Full State Compensator
def compensator(A, B, C, D, L, x0, x_hat_0, u, dt):
    x_hat = x_hat_0
    x = np.transpose(np.array([x0]))

    # State Feedback
    y = C@x + D@u
    x_dot = A@x + B@u    
    x = x + x_dot*dt

    # STate Estimator
    x_hat_dot = A@x_hat + B@u + L@(y-C@x_hat)
    x_hat = x_hat + x_hat_dot*dt

    return x, x_hat

#---------------- Set up the Simulation!---------------------
# get environment
# start time to count the computation
prev_time = pygame.time.get_ticks()
env = gym.make("CartPole-v1", render_mode="human")

# Getting Observation of the cart pole
obs = np.array(env.reset()[0])
print('\nGet initial observation of the cartpole is:', obs)

reward_total = 0

for i in range(1000):
    env.render()
    
    # get force direction (action) and force value (force)
    action, force = apply_state_controller(K, x_hat)

    # apply compensator function
    x, x_hat = compensator(A, B, C, D, L, obs, x_hat, force, dt)      

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -10, 10)))

    # change magnitute of the applied force in CartPole
    env.env.force_mag = abs_force

    # apply action
    obs, reward, done, _, _ = env.step(action)  
    
    reward_total = reward_total+reward    
    
    # geting value of Pole Angle
    tetha = x[2]

    # Calculating the needed time in system
    curr_time = pygame.time.get_ticks()
    dt = (curr_time - prev_time)/1000
    prev_time = curr_time
    
    #Input element to the list 
    x_array.append(x)    
    x_hat_array.append(x_hat)    
    i_array.append(i) 
    t_array.append((curr_time/1000))
    u_array.append(abs_force)
    tetha_array.append(tetha*0.0174533) #Convert Degree to radian

    if done or reward_total == 200:
        print(f'\nTerminated after {i+1} iterations.')
        print(f"Total Reward = {reward_total}")
        break
env.close()

# calculate the average of the force
print(f"Force Average = {sum(u_array)/len(u_array):.5f} Newton")
print(f"Value SSE of Theta = {float(max(tetha_array)):.5f} Radian")

#------------Plotting the system response------------------
#Convert list to the array
x_array = np.array(x_array)
x_hat_array = np.array(x_hat_array)
i_array = np.array(i_array)
t_array = np.array(t_array)

subplots = []
datapoints, state_num, _ = x_array.shape
style.use('seaborn-dark')

fig, ax = plt.subplots(state_num, sharex=True, sharey=True)
fig.suptitle("System Response with Step by Iterations")

for i in range(0, state_num):
  ax[i].plot(i_array[:], x_array[:, i], '-r', lw = 1)
  ax[i].plot(i_array[:], x_hat_array[:, i], '--b', lw = 1)    
  ax[i].grid()

plt.show()