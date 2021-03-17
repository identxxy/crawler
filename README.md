# Intro

This is a simple 4-leg robot simulation model and integrated with reinforcement learning algorithm. Each of its leg has 4 DOF. The simulation platform is [Gazebo](http://gazebosim.org/) using [ROS](https://www.ros.org/) control and [openai_ros](http://wiki.ros.org/openai_ros) to integrate the model with the RL framework [Gym](https://gym.openai.com/).

![screenshot](screenshot.png)

# Environment Setup

- ubuntu 20.04. To set up the system with only python3 `sudo apt install python-is-python3`
- ROS-Noetic ([Installation guide](http://wiki.ros.org/noetic/Installation/Ubuntu))
    - Gazebo (contained in the `ros-noetic-desktop-full` in the above guide)
    - effort controller `sudo apt install ros-noetic-effort-controller`
    - velocity controller `sudo apt install ros-noetic-velocity-controller`
    - position controller `sudo apt install ros-noetic-position-controller`
- Gym version 0.17.2 `pip3 install gym`
- Torch version 1.7.1 `pip3 install torch`

Remember follow [this guide](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment) to create a ROS workspace. e.g. `~/catkin_ws`

Clone the repo and move it to `~/catkin_ws/src`. Then run `catkin_make` at `~/catkin_ws`

# Usage

Firstly, run the simulation, cmd in `[]` is optional  
`roslaunch crawler world.launch [gui:=false]`

Then run the training script in the `script` dir.  
`python crawlerD.py`