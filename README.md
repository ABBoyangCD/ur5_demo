# ur5-demo

## Scripts
* **mask.py** MaskRCNN takes in an image, outputs masks, boxes, labels and segmented image.
* **talker.py**

## Full instalation procedure
0. Install ROS Noetic:

```bash
wget -c https://raw.githubusercontent.com/qboticslabs/ros_install_noetic/master/ros_install_noetic.sh && chmod +x ./ros_install_noetic.sh && ./ros_install_noetic.sh
```

1. Setup the workspace:

```bash
mkdir -p ur5_ws/src && cd ur5_ws
```

2. Setup dependencies:

```bash
sudo apt -y install ros-noetic-moveit \
  ros-noetic-realsense2-camera \
  ros-noetic-realsense2-description
```

3. Get the UR-drivers

```bash
# source global ros
source /opt/ros/noetic/setup.bash

# clone the driver
git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git src/Universal_Robots_ROS_Driver

# clone fork of the description. This is currently necessary, until the changes are merged upstream.
# git clone -b calibration_devel https://github.com/fmauch/universal_robot.git src/fmauch_universal_robot
git clone -b melodic-devel-staging https://github.com/ros-industrial/universal_robot.git src/universal_robot

# get moveit calibration
git clone git@github.com:ros-planning/moveit_calibration.git src/moveit_calibration

# get ur5-demo package

# get the demo
git clone git@github.com:tudorjnu/ur5-demo.git src/ur5_demo

# install dependencies
sudo apt update -qq
rosdep update
rosdep install --from-paths src --ignore-src -y --rosdistro noetic

# build the workspace
catkin_make

# activate the workspace (ie: source it)
echo "source devel/setup.bash" >> ~/.bashrc
```

