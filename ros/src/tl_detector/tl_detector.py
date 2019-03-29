#!/usr/bin/env python

import yaml
from math import sqrt

import rospy
import tf
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from styx_msgs.msg import Lane
from styx_msgs.msg import TrafficLightArray, TrafficLight

from light_classification.tl_classifier import TLClassifier

LOG_LEVEL = rospy.INFO



class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector', log_level=LOG_LEVEL)

        config_string = rospy.get_param("/traffic_light_config")

        self.config = yaml.load(config_string)

        self.pose = None
        self.closest_wp_idx = None
        self.waypoints = None
        self.waypoints_len = None
        self.waypoints_last_idx = None
        self.camera_image = None
        self.state = None

        self.state_list = [None, None, None]
        self.tl_wp_indices = None
        self.traffic_lights = None

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)        
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=5)
        sub7 = rospy.Subscriber('/closest_wp_idx_2V', Int32, self.closest_wp_idx_2V_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=5)

        self.loop()

    def loop(self):
        rate = rospy.Rate(6) # 6 Hz
        while not rospy.is_shutdown():
            # Process and publish the traffic light state
            self.process_traffic_lights()
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg.pose

    def waypoints_cb(self, msg):
        self.waypoints = msg.waypoints
        self.waypoints_len = len(self.waypoints)
        self.waypoints_last_idx = self.waypoints_len - 1

    def traffic_cb(self, msg):
        self.traffic_lights = msg.lights        

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.camera_image = msg

    def closest_wp_idx_2V_cb(self, msg):
        self.closest_wp_idx = msg.data

    def get_closest_waypoint(self, position):
        min_dist = 99999999.9
        closest_wp_idx = -1

        for i, wp in enumerate(self.waypoints):
            dist = sqrt((wp.pose.pose.position.x - position.x) ** 2 +
                        (wp.pose.pose.position.y - position.y) ** 2)
            if dist < min_dist:
                closest_wp_idx, min_dist = i, dist

        return closest_wp_idx

    def get_light_state(self):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if self.camera_image is not None:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """
        Finds closest visible traffic light, if one exists, and determines its 
        location and color and index of waypoint closes to the upcoming stop 
        line for a traffic light 
        
        Publish the state of the traffic light
        """
        if self.closest_wp_idx is not None and \
                self.traffic_lights is not None and \
                self.waypoints is not None:
            next_tl_wp_idx = None
            dist_to_next_tl = None
            temp_min_idx = 9999999999
            for tl_idx, tl in enumerate(self.traffic_lights):
                stop_line_pos = Point()
                stop_line_pos.x = self.config["stop_line_positions"][tl_idx][0]
                stop_line_pos.y = self.config["stop_line_positions"][tl_idx][1]

                tl_wp_idx = self.get_closest_waypoint(stop_line_pos)
                if tl_wp_idx > self.closest_wp_idx and tl_wp_idx < temp_min_idx:
                    next_tl_wp_idx = tl_wp_idx
                    temp_min_idx = next_tl_wp_idx

            if next_tl_wp_idx is not None:
                next_tl_wp = self.waypoints[next_tl_wp_idx]
                dist_to_next_tl = sqrt((next_tl_wp.pose.pose.position.x - self.pose.position.x) ** 2 +
                                       (next_tl_wp.pose.pose.position.y - self.pose.position.y) ** 2)

                next_traffic_light = self.get_light_state()
                self.state_list.append(next_traffic_light)
                self.state_list.pop(0)
                # Check if all elements in the state list are the same
                if len(set(self.state_list)) == 1 and next_traffic_light != self.state:
                    self.state = next_traffic_light     

                # Publish the traffic light state in 70m distance
                if self.state == TrafficLight.RED and dist_to_next_tl < 50:
                    self.upcoming_red_light_pub.publish(Int32(next_tl_wp_idx))
                else:
                    self.upcoming_red_light_pub.publish(Int32(-1)) # -1 for no valid red light



if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
