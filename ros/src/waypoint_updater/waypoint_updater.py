#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
import math
from scipy.spatial import KDTree
import numpy as np



'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOG_LEVEL = rospy.INFO


MAX_SPEED = 40.0 * 0.27778  # m/s
MAX_DECC = 10.0  # m/s^2
LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
DT_BTW_WPS = 0.02  # Time interval s



class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=LOG_LEVEL)



        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)
        # Publish the index of the closest waypoint to the vehicle
        self.closest_wp_idx_2V_pub = rospy.Publisher('/closest_wp_idx_2V', Int32, queue_size=1)

        # Add other member variables you need below
        self.current_pose = None
        self.current_velocity = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.base_waypoints_len = None
        self.base_waypoints_last_idx = None
        self.current_closest_wp_idx = None
        self.stopline_wp_idx = None
        self.dec_start_wp_idx = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.publish_final_waypoints()
            rate.sleep()

    def pose_cb(self, msg):
        self.current_pose = msg


    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist

    def waypoints_cb(self, msg):
        self.base_waypoints = msg.waypoints
        self.base_waypoints_len = len(self.base_waypoints)
        self.base_waypoints_last_idx = self.base_waypoints_len - 1
        if self.waypoints_2d is not True:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in msg.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)


    def traffic_cb(self, msg):
        if msg.data == -1: # No red traffic light
            self.stopline_wp_idx = None
            self.dec_start_wp_idx = None
            rospy.loginfo("no red traffic light detected, let's go")
        else: # Red traffic light detected
            self.stopline_wp_idx = msg.data
            n_wp_to_stop = math.ceil(MAX_SPEED/MAX_DECC/DT_BTW_WPS)
            self.dec_start_wp_idx = self.stopline_wp_idx - n_wp_to_stop
            rospy.loginfo("red traffic light info received, wp idx=%d", self.stopline_wp_idx)

    def obstacle_cb(self, msg):
        pass

    def publish_final_waypoints(self):
        next_lane = self.generate_lane_object()
        self.final_waypoints_pub.publish(next_lane)

    def get_closest_wp_idx(self):
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y

        closest_idx = self.waypoint_tree.query([x,y], 1)[1]
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        cl_vect = np.array(closest_coord)
        pos_vect = np.array([x,y])
        val = np.dot(cl_vect - prev_coord, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx


    def distance(self, pt1, pt2):
        dx = pt1.x - pt2.x
        dy = pt1.y - pt2.y
        return math.sqrt(dx*dx + dy*dy)

    def generate_lane_object(self):
        lane = Lane()
        if self.current_pose is not None and self.base_waypoints is not None:
            closest_wp_idx = self.get_closest_wp_idx()
            self.current_closest_wp_idx = closest_wp_idx
            # Publish the closest way point index to /closest_wp_idx_2V topic
            self.closest_wp_idx_2V_pub.publish(Int32(self.current_closest_wp_idx))
            next_wps = []
            next_wps_idx_start = closest_wp_idx
            next_wps_idx_end = min(self.base_waypoints_last_idx, closest_wp_idx + LOOKAHEAD_WPS - 1)
            rospy.logdebug("next final wp idx: %s-%s", next_wps_idx_start, next_wps_idx_end)
            for i in range(next_wps_idx_start, next_wps_idx_end + 1):
                wp = Waypoint()
                wp.pose.pose.position.x = self.base_waypoints[i].pose.pose.position.x
                wp.pose.pose.position.y = self.base_waypoints[i].pose.pose.position.y
                wp.pose.pose.position.z = self.base_waypoints[i].pose.pose.position.z
                wp.twist.twist.linear.x = self.base_waypoints[i].twist.twist.linear.x
                next_wps.append(wp)

            if len(next_wps) < 2:
                for waypoint in next_wps:
                    waypoint.twist.twist.linear.x = 0
            else: # Deceleration for red traffic light
                if self.stopline_wp_idx >= 0 and self.dec_start_wp_idx >= 0:
                    next_wps = self.decelerate_waypoints(next_wps, next_wps_idx_start, next_wps_idx_end)
            lane.waypoints = next_wps
        return lane

    def decelerate_waypoints(self, next_wps, next_wps_idx_start, next_wps_idx_end):
        if self.dec_start_wp_idx > next_wps_idx_end:
            return next_wps
        else:
            for wp_idx, wp_pos_idx in enumerate(range(next_wps_idx_start, next_wps_idx_end)):
                expected_speed = MAX_SPEED - MAX_DECC * (wp_pos_idx - self.dec_start_wp_idx) * DT_BTW_WPS * 1.1
                next_wps[wp_idx].twist.twist.linear.x = min(expected_speed, next_wps[wp_idx].twist.twist.linear.x)
            return next_wps



if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
