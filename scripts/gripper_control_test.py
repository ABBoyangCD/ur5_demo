#!/usr/bin/env python3
from __future__ import print_function
import time
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg
import rospy

import roslib
roslib.load_manifest('robotiq_2f_gripper_control')


def genCommand(char):
    """Update the command according to the character entered by the user."""

    command = outputMsg.Robotiq2FGripper_robot_output()

    if char == 'a':
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


if __name__ == '__main__':

    """Main loop which requests new commands and publish them on the Robotiq2FGripperRobotOutput topic."""
    rospy.init_node('Robotiq2FGripperSimpleController')

    pub = rospy.Publisher('Robotiq2FGripperRobotOutput',
                          outputMsg.Robotiq2FGripper_robot_output, queue_size=10)
    try:
        while not rospy.is_shutdown():

            # Send the command to the gripper
            pub.publish(genCommand("r"))
            time.sleep(0.1)
            pub.publish(genCommand("a"))
            # Sleep to give I/O time
            time.sleep(3)

    except KeyboardInterrupt:
        print("Shutting down")
