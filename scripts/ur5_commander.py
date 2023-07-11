#!/usr/bin/env python3

from __future__ import print_function
import time
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg
import rospy
import roslib
import sys
import argparse
import random
import moveit_commander
import moveit_msgs.msg
from moveit_msgs.msg import RobotTrajectory, Grasp, PlaceLocation
from geometry_msgs.msg import PoseStamped
import tf

roslib.load_manifest('robotiq_2f_gripper_control')

# rpy to quaternion


def rpy_to_quaternion(roll, pitch, yaw):
    q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    return q


def gen_command(char):
    """Update the command according to the character entered by the user."""

    command = outputMsg.Robotiq2FGripper_robot_output()

    if char == 'a':
        print("Activating the gripper")
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rACT = 1
        command.rGTO = 1
        command.rSP = 255
        command.rFR = 150

    if char == 'r':
        print("Resetting...")
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rACT = 0

    if char == 'c':
        command.rPR = 255

    if char == 'o':
        command.rPR = 0

    # If the command entered is a int, assign this value to rPRA
    try:
        command.rPR = int(char)
        if command.rPR > 255:
            command.rPR = 255
        if command.rPR < 0:
            command.rPR = 0
    except ValueError:
        pass

    if char == 'f':
        command.rSP += 25
        if command.rSP > 255:
            command.rSP = 255

    if char == 'l':
        command.rSP -= 25
        if command.rSP < 0:
            command.rSP = 0

    if char == 'i':
        command.rFR += 25
        if command.rFR > 255:
            command.rFR = 255

    if char == 'd':
        command.rFR -= 25
        if command.rFR < 0:
            command.rFR = 0

    return command


class UR5:

    def __init__(self,
                 move_group_name: str = "manipulator",
                 reference_frame: str = "base_link",
                 end_effector_link: str = None,
                 robot_description: str = "robot_description",
                 planning_time: float = 20.0,
                 num_planning_attempts: int = 100,
                 goal_tolerance: float = 0.01,
                 goal_orientation_tolerance: float = 0.01,
                 max_velocity_scaling_factor: float = 0.1,
                 max_acceleration_scaling_factor: float = 0.1,
                 max_pick_attempts: int = 10,
                 go_home: bool = False,
                 gripper_length: float = 0.2):

        self.gripper_pub = rospy.Publisher('Robotiq2FGripperRobotOutput',
                                           outputMsg.Robotiq2FGripper_robot_output, queue_size=10)
        self.gripper_pub.publish(gen_command('r'))
        rospy.sleep(1)
        self.base_orientation = [0.0419532, -0.017782, 0.0330182, 0.9984155]

        # set gripper parameters
        self.gripper_length = gripper_length
        self.pre_grasp_approach = 0.05
        self.post_grasp_retreat = 0.05

        self.max_pick_attempts = max_pick_attempts

        # Initialize the move group for the ur5_arm
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        self.group = moveit_commander.MoveGroupCommander(move_group_name)
        # for displaying the trajectory in rviz
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20)

        # set the group parameters
        self.group.set_planning_time(planning_time)
        self.group.set_num_planning_attempts(num_planning_attempts)
        self.group.set_goal_position_tolerance(goal_tolerance)
        self.group.set_goal_orientation_tolerance(goal_orientation_tolerance)
        self.group.set_max_velocity_scaling_factor(max_velocity_scaling_factor)
        self.group.set_max_acceleration_scaling_factor(
            max_acceleration_scaling_factor)
        self.group.set_planner_id("RRTstark")
        self.group.allow_replanning(True)
        self.group.set_pose_reference_frame(reference_frame)

        if end_effector_link is not None:
            self.group.set_end_effector_link(end_effector_link)

        # check all the objects in the scene
        objects = self.scene.get_known_object_names()
        # clear all objects in the scene
        for obj in objects:
            self.scene.remove_world_object(obj)

        self.place_pose = [0.17, -1.27,
                           1.22, 1.63,
                           1.5708, -1.5708]

        # table parameters
        table_height = 0.00
        table_size = [1.5, 1.5, 0.01]
        table_orientation = [0, 0, 0, 1]
        table_position = [0, 0, table_height - table_size[2] / 2 - 0.001]

        # add table to the scene
        table_pose = self.list_to_pose(table_position, table_orientation)
        self.add_object(table_pose, table_size, "table")

        # set the workspace dimensions to the table size
        self.group.set_workspace([-table_size[0] / 2, table_size[0] / 2,
                                  -table_size[1] / 2, table_size[1] / 2,
                                  0.05, 1.5])

        # set table as the support surface
        self.group.set_support_surface_name("table")

        # place PoseStamped
        # self.place_pose = self.list_to_pose(
        # [0.5, 0.5, table_height + 0.01], [0, 0, 0, 1])

        # get end effector pose
        self.end_effector_pose = self.group.get_current_pose()

        # set target position to the named target in the SRDF
        if go_home:
            self.group.set_named_target("home")
            self.group.go()

            # clear everything
            self.group.stop()
            self.group.clear_pose_targets()
            rospy.sleep(2)

        print("Fully initialized")

    def get_position(self):
        position = self.group.get_current_pose().pose.position
        orientation = self.group.get_current_pose().pose.orientation
        return (position, orientation)

    # move to pose
    def move_to_pose(self,
                     pose: PoseStamped):
        self.group.set_pose_target(pose)
        plan = self.group.plan()[1]
        self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        rospy.sleep(2)

    def move_to(self,
                target_pose: PoseStamped):
        self.group.set_start_state_to_current_state()

        self.group.set_pose_target(target_pose)
        plan = self.group.plan()[1]
        while True:
            self.display_plan(plan)
            ans = input("Execute plan? (y/n): ")
            if ans == "y":
                self.group.execute(plan, wait=True)
                break
            elif ans == "n":
                plan = self.group.plan()[1]
        self.group.stop()
        self.group.clear_pose_targets()
        rospy.sleep(2)

    # add box to the scene
    def add_object(self,
                   pose: PoseStamped,
                   size: list,
                   object_id: str = "object"):

        print(f"...Adding {object_id}...")
        # pose.header.frame_id = self.group.get_planning_frame()
        self.scene.add_box(object_id, pose, size)
        print(f"...Added {object_id}...")

    # display a plan in Rviz
    def display_plan(self,
                     plan: RobotTrajectory):
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        self.display_trajectory_publisher.publish(display_trajectory)

    # convert from a tuple of position and orientation to PoseStamped
    def list_to_pose(self,
                     position: list,
                     orientation: list):

        pose = PoseStamped()
        pose.header.frame_id = self.group.get_planning_frame()
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        pose.pose.orientation.x = orientation[0]
        pose.pose.orientation.y = orientation[1]
        pose.pose.orientation.z = orientation[2]
        pose.pose.orientation.w = orientation[3]

        return pose

    # generate a list of possible grasps given a PoseStamped
    def generate_grasps(self,
                        target_pose: PoseStamped):

        grasps = Grasp()

        # set the grasp position
        grasps.grasp_pose.pose.position = target_pose.pose.position
        # self.end_effector_pose.pose.orientation
        grasps.grasp_pose.pose.orientation = target_pose.pose.orientation

        # set the pre-grasp approach
        grasps.pre_grasp_approach.direction.header.frame_id = self.group.get_planning_frame()
        grasps.pre_grasp_approach.direction.vector.z = 1.0
        grasps.pre_grasp_approach.min_distance = 0.02
        grasps.pre_grasp_approach.desired_distance = 0.05

        # set the post-grasp retreat
        grasps.post_grasp_retreat.direction.header.frame_id = self.group.get_planning_frame()
        grasps.post_grasp_retreat.direction.vector.z = 1.0
        grasps.post_grasp_retreat.min_distance = 0.02
        grasps.post_grasp_retreat.desired_distance = 0.05

        return grasps

    def open_gripper(self):
        self.gripper_pub.publish(gen_command("o"))
        rospy.sleep(3)

    def close_gripper(self):
        self.gripper_pub.publish(gen_command("c"))
        rospy.sleep(3)

    @staticmethod
    def quaternion_from_euler(roll, pitch, yaw):
        q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        return q

    @staticmethod
    def euler_from_quaternion(q):
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(q)
        return [roll, pitch, yaw]

    def go_to_place(self):
        target = self.place_pose
        self.group.set_joint_value_target(target)
        plan = self.group.plan()[1]
        while True:
            self.display_plan(plan)
            ans = input("Execute plan? (y/n): ")
            if ans == "y":
                self.group.execute(plan, wait=True)
                break
            elif ans == "n":
                plan = self.group.plan()[1]
        self.group.stop()
        self.group.clear_pose_targets()
        rospy.sleep(2)


def pose_generator(orientation: list = [0.1745071, 0.9846534, 0.0021049, 0.000709]) -> PoseStamped:
    x = random.uniform(-0.5, 0.5)
    y = random.uniform(-0.5, 0.5)
    z = random.uniform(0.1, 0.5)

    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z

    pose.pose.orientation.x = orientation[0]
    pose.pose.orientation.y = orientation[1]
    pose.pose.orientation.z = orientation[2]
    pose.pose.orientation.w = orientation[3]

    return pose


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", type=str, choices=["real", "sim"],
                    required=False, default="real", help="Select mode: real or sim")
    ap.add_argument("--home", type=bool,
                    required=False, default=False, help="Go to home position")
    args = vars(ap.parse_args())

    rospy.init_node("ur5_moveit", anonymous=True)

    pre_grasp_pub = rospy.Publisher(
        "/ur5/pre_grasp_pose", PoseStamped, queue_size=1)

    grasp_pub = rospy.Publisher(
        "/ur5/grasp_pose", PoseStamped, queue_size=1)

    target_pub = rospy.Publisher(
        "/ur5/target_pose", PoseStamped, queue_size=1)

    if args["mode"] == "real":
        ur5 = UR5(reference_frame="camera_color_optical_frame",
                  end_effector_link="wrist_3_link",
                  go_home=False)
    elif args["mode"] == "sim":
        ur5 = UR5(reference_frame="base_link", go_home=args["home"])

    def main(target_pose):

        orientation = ur5.quaternion_from_euler(0, 0, -3.14)
        target_pose.pose.orientation.x = orientation[0]
        target_pose.pose.orientation.y = orientation[1]
        target_pose.pose.orientation.z = orientation[2]
        target_pose.pose.orientation.w = orientation[3]

        # Send the command to the gripper
        # set the gripper
        # ur5.gripper_pub.publish(gen_command("r"))
        rospy.sleep(1)
        # ur5.gripper_pub.publish(gen_command("a"))
        # Sleep to give I/O time
        rospy.sleep(3)

        # remove objects from the scene

        # pre-grasp Pose
        grasp_pose = target_pose
        grasp_pose.pose.position.z -= ur5.gripper_length

        # grasps = ur5.generate_grasps(grasp_pose)

        ur5.scene.remove_world_object("object")
        # rospy.sleep(1)
        # ur5.add_object(grasp_pose, [0.04, 0.04, 0.04], "object")
        # rospy.sleep(1)
        # plan = ur5.group.pick("object", grasps, plan_only=True)

        # while True:
        # execute = input("Execute plan?")
        # if execute == "y":
        # ur5.group.execute(plan, wait=True)
        # else:
        # plan = ur5.group.pick("object", grasps, plan_only=True)
        # break

        pre_grasp_pose = grasp_pose
        # pre_grasp_pose.pose.position.z -= 0.02

        pre_grasp_pub.publish(pre_grasp_pose)
        grasp_pub.publish(grasp_pose)
        target_pub.publish(target_pose)

        ur5.move_to(pre_grasp_pose)
        rospy.sleep(2)
        ur5.move_to(grasp_pose)

        # press enter to continue
        input("Press enter to close gripper")
        # ur5.close_gripper()

        ur5.go_to_place()

        # clear everything
        ur5.group.stop()
        ur5.group.clear_pose_targets()
        rospy.sleep(2)

        drop = input("Drop object? y/n")
        if drop == "y":
            ur5.open_gripper()
            rospy.sleep(2)

    try:
        if args["mode"] == "real":
            rospy.Subscriber("/vision/pose", PoseStamped, main)
            rospy.spin()
        elif args["mode"] == "sim":
            main(pose_generator())
            rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
