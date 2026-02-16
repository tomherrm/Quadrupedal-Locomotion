# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

""" Run CPG """

import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states

#peut etre il faudrait initaliser dautres trucs ?

q=np.zeros(12)
dq=np.zeros(12)

rvalues = np.zeros((TEST_STEPS,4))  # amplitude for plotting
drvalues = np.zeros((TEST_STEPS,4))
thetavalues = np.zeros((TEST_STEPS,4))
dthetavalues = np.zeros((TEST_STEPS,4))

desiredpos= np.zeros((TEST_STEPS,3))
realpos= np.zeros((TEST_STEPS,3))

desiredangles= np.zeros((TEST_STEPS,3))
realangles= np.zeros((TEST_STEPS,3))



############## Sample Gains
# joint PD gains
kp=np.array([100,100,100]) #TROT,PACE: np.array([100,100,100])
kd=np.array([2,2,2])#BOUND:np.array([3,3,3]) 

# Cartesian PD gains
kpCartesian = np.diag([500]*3) #TROT,PACE:np.diag([500]*3)
kdCartesian = np.diag([20]*3)#TROT,PACE:np.diag([20]*3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 

  # get desired foot positions from CPG 
  xs,zs = cpg.update()

  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)

    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])

    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(i,leg_xyz) # [TODO] #les joint angles desires

    # Add joint PD contribution to tau for leg i (Equation 4)
    tau += kp*(leg_q-q[3*i:3*i+3])+kd*(0-dq[3*i:3*i+3]) # [TODO] 

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get desired xyz position in leg frame (use ComputeJacobianAndPosition with the joint angles you just found above)
      # [TODO]

      J,current_xyz=env.robot.ComputeJacobianAndPosition(i,q[3*i:3*i+3]) #jai mis : the function :Can also get the Jacobian / foot position for a specific joint angle configuration.

      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      # [TODO] 

      #ca sert a quoi ca ??

      # Get current foot velocity in leg frame (Equation 2)
      # [TODO] 

      current_footv=J@dq[3*i:3*i+3]

      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau += J.T@(kpCartesian@(leg_xyz-current_xyz)+kdCartesian@(0-current_footv)) # [TODO]

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau
    #print(tau)

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

  # [TODO] save any CPG or robot states
  #print(cpg.get_r())
  rvalues[j, :] = cpg.get_r()
  drvalues[j, :] = cpg.get_dr()
  thetavalues[j, :] = cpg.get_theta()
  dthetavalues[j, :] = cpg.get_dtheta()

  desiredpos[j,:] =np.array([xs[0], sideSign[0] * foot_y, zs[0]])
  _,realpos[j,:]=env.robot.ComputeJacobianAndPosition(0,q[0:3])

  desiredangles[j,:]=env.robot.ComputeInverseKinematics(0,desiredpos[j,:])
  realangles[j,:]=q[0:3]
##################################################### 
# PLOTS
#####################################################
# [TODO] Create your plots

# example
# fig = plt.figure()
# plt.plot(t,joint_pos[1,:], label='FR thigh')
# plt.legend()
# plt.show()
#print(rvalues)

plotdesiredpos=True

plotdesiredang=True

plotampliandphase=True

if plotdesiredang:
  fig=plt.figure()
  plt.plot(t,desiredangles[:,0], label='desired q1')
  plt.plot(t,desiredangles[:,1], label='desired q2')
  plt.plot(t,desiredangles[:,2], label='desired q3')
  plt.plot(t,realangles[:,0], label='real q1')
  plt.plot(t,realangles[:,1], label='real q2')
  plt.plot(t,realangles[:,2], label='real q3')
  plt.legend()
  plt.show()


if plotdesiredpos :
  fig=plt.figure()
  plt.plot(t,desiredpos[:,0], label='desired x')
  plt.plot(t,desiredpos[:,1], label='desired y')
  plt.plot(t,desiredpos[:,2], label='desired z')
  plt.plot(t,realpos[:,0], label='real x')
  plt.plot(t,realpos[:,1], label='real y')
  plt.plot(t,realpos[:,2], label='real z')
  plt.legend()
  plt.show()



if plotampliandphase :
  fig, axs = plt.subplots(4, 2, figsize=(12, 8))  # 2x2 grid

  #FR
  axs[0,0].plot(t,rvalues[:,0], label='r')
  axs[0,0].plot(t,drvalues[:,0], label='dr')
  axs[0,0].set_title('FR Amplitudes')

  axs[0,0].legend()
  axs[0,0].grid(True)

  axs[0,1].plot(t,thetavalues[:,0], label='theta')
  axs[0,1].plot(t,dthetavalues[:,0], label='dtheta')
  axs[0,1].set_title('FR Phases')

  axs[0,1].legend()
  axs[0,1].grid(True)

  #FL
  axs[1,0].plot(t,rvalues[:,1], label='r')
  axs[1,0].plot(t,drvalues[:,1], label='dr')
  axs[1,0].set_title('FL Amplitudes')

  axs[1,0].legend()
  axs[1,0].grid(True)

  axs[1,1].plot(t,thetavalues[:,1], label='theta')
  axs[1,1].plot(t,dthetavalues[:,1], label='dtheta')
  axs[1,1].set_title('FL Phases')

  axs[1,1].legend()
  axs[1,1].grid(True)

  #RR
  axs[2,0].plot(t,rvalues[:,2], label='r')
  axs[2,0].plot(t,drvalues[:,2], label='dr')
  axs[2,0].set_title('RR Amplitudes')

  axs[2,0].legend()
  axs[2,0].grid(True)

  axs[2,1].plot(t,thetavalues[:,2], label='theta')
  axs[2,1].plot(t,dthetavalues[:,2], label='dtheta')
  axs[2,1].set_title('RR Phases')

  axs[2,1].legend()
  axs[2,1].grid(True)

  #RL
  axs[3,0].plot(t,rvalues[:,3], label='r')
  axs[3,0].plot(t,drvalues[:,3], label='dr')
  axs[3,0].set_title('RL Amplitudes')

  axs[3,0].legend()
  axs[3,0].grid(True)

  axs[3,1].plot(t,thetavalues[:,3], label='theta')
  axs[3,1].plot(t,dthetavalues[:,3], label='dtheta')
  axs[3,1].set_title('RL Phases')

  axs[3,1].legend()
  axs[3,1].grid(True)

  plt.tight_layout()
  plt.show()


