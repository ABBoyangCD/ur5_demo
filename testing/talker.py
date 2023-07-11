#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose

# create a pose message


def create_pose_msg(position, orientation):
    pose = PoseStamped()
    pose.pose.position.x = position[0]
    pose.pose.position.y = position[1]
    pose.pose.position.z = position[2]
    pose.pose.orientation.x = orientation[0]
    pose.pose.orientation.y = orientation[1]
    pose.pose.orientation.z = orientation[2]
    pose.pose.orientation.w = orientation[3]
    return pose


def talker():
    pub = rospy.Publisher('/ur5/target_pose', PoseStamped, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(0.1)
    while not rospy.is_shutdown():
        # make random x y z position with values between -0.4 and 0.4 except the interval [-0.2, 0.2]
        x = np.random.uniform(-0.4, 0.4)
        while abs(x) < 0.2:
            x = np.random.uniform(-0.4, 0.4)
        y = np.random.uniform(-0.4, 0.4)
        while abs(y) < 0.2:
            y = np.random.uniform(-0.4, 0.4)
        z = np.random.uniform(0.1, 0.12)

        pose = create_pose_msg([x, y, z], [0.0, 0.0, 0.0, 1.0])
        rospy.loginfo(pose)
        pub.publish(pose)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
