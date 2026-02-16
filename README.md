
 # Quadrupedal Locomotion via CPG-guided Reinforcement Learning 
This repository contains the implementation of a quadrupedal locomotion system that integrates Central Pattern Generators (CPG) with Deep Reinforcement Learning (DRL) to achieve robust gaits on complex terrains.

## 🚀 Project Overview
 

Rhythmic Control: Implements a network of coupled Hopf oscillators to generate stable, periodic limit cycles for leg movement.


 Gait Diversity: Supports four fundamental gaits—Trot, Walk, Pace, and Bound—defined by specific inter-oscillator phase-lags.



 Sim-to-Real Focus: Utilizes curriculum-based domain randomization to bridge the gap between simulation and hardware execution.

## 🛠️ Methodology


### Reinforcement Learning Pipeline

Algorithm: Employs Proximal Policy Optimization (PPO) for high sample efficiency and stability compared to Soft Actor-Critic (SAC).



Observation Space: Fuses proprioceptive data (joint angles, base orientation, foot contacts) with internal CPG states (phase and amplitude) to provide a structured rhythmic prior.



Reward Engineering: Features a multi-objective reward function balancing velocity tracking, orientation stability, and energy efficiency (Cost of Transport).


### Control Architectures

CPG Action Space: Constrains the policy to output rhythmic parameters (frequency and amplitude), resulting in smoother, more physically plausible gaits.



Cartesian PD Control: Implements task-space corrections that reduce end-effector tracking error by 8.9% compared to joint-space control alone.


Hierarchical Training: Uses Adaptive Curricular Training to progressively increase terrain difficulty, enabling the robot to surmount inclines of up to 29%.

## 📊 Key Results


Velocity Range: Achieved controlled forward speeds from a minimum of 0.23 m/s to a maximum of 1.53 m/s.


Efficiency: Demonstrated that Cost of Transport (CoT) decreases as velocity increases due to improved dynamic stability at higher speeds.


Ablation Studies: Performed extensive testing on neural network architectures, finding that a 2-layer [256, 256] network offered superior reliability for slope traversal over deeper models.
