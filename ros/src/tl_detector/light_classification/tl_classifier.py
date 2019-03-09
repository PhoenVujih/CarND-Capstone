from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import os
import cv2
import rospy
import yaml

RECORD_IMAGES = False


class TLClassifier(object):
    def __init__(self):
        self.frozen_graph = None
        self.session = None
        self.image_counter = 0
        # Note: These class numbers match those defined in label_map.pbtxt used in training done by tensorflow object
        # detection api. They are DIFFERENT than those defined in ROS message TrafficLight.msg. Conversion needs to be
        # done here before reporting back to tl_detector.
        self.classes = {1: TrafficLight.GREEN,
                        2: TrafficLight.RED,
                        3: TrafficLight.YELLOW,
                        4: TrafficLight.UNKNOWN}

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        if self.config['is_site'] is True:
            model_path = 'light_classification/model/real/frozen_inference_graph.pb'
        else:
            model_path = 'light_classification/model/sim/frozen_inference_graph.pb'
        
        self.frozen_graph = tf.Graph()
        with tf.Session(graph=self.frozen_graph) as sess:
            self.session = sess
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        class_index, probability = self.inference(image)

        if class_index is not None:
            rospy.logdebug("class: %d, probability: %f", class_index, probability)

        return class_index

    def inference(self, image, threshold=0.5):
        image_tensor = self.frozen_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self.frozen_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.frozen_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.frozen_graph.get_tensor_by_name('detection_classes:0')
        
        image = cv2.cvtColor(cv2.resize(image, (300, 300)), cv2.COLOR_BGR2RGB)

        (boxes, scores, classes) = self.session.run(
            [detection_boxes, detection_scores, detection_classes],
            feed_dict={image_tensor: np.expand_dims(image, axis=0)})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        boxes = np.squeeze(boxes)

        for i, box in enumerate(boxes):
            if scores[i] > threshold:
                light_class = self.classes[classes[i]]
                rospy.logdebug("Traffic Light Class detected: %d", light_class)
                return light_class, scores[i]

        return None, None