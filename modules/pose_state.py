import math

import numpy as np

from modules.pose import Pose


def get_paf(current_pose, joint):
    start = joint['start']
    end = joint['end']
    start = np.array(current_pose.keypoints[start])
    end = np.array(current_pose.keypoints[end])

    paf = end - start
    deviation = math.inf if np.any(start < 0) or np.any(end < 0) else current_pose.confidence + current_pose.confidence

    return paf, deviation


def unit_vector(a):
    return a / np.linalg.norm(a)


def angle(u, v):
    unit_u = unit_vector(u)
    unit_v = unit_vector(v)
    return np.arccos(np.clip(np.dot(unit_u, unit_v), -1.0, 1.0))


class PoseState:
    def __init__(self, pose_json):
        self.name = pose_json['name']
        self.message = pose_json['message']
        self.corrections = []
        key_point_names = Pose.kpt_names
        # print(key_point_names)
        for correction_json in pose_json['corrections']:
            correction = dict()
            correction['joint_1'] = {}
            correction['joint_1']['start'] = key_point_names.index(correction_json['joint_1']['start'])
            correction['joint_1']['end'] = key_point_names.index(correction_json['joint_1']['end'])
            correction['joint_2'] = {}
            correction['joint_2']['start'] = key_point_names.index(correction_json['joint_2']['start'])
            correction['joint_2']['end'] = key_point_names.index(correction_json['joint_2']['end'])
            correction['angle'] = correction_json['angle']
            correction['deviation'] = correction_json['deviation']
            correction['message'] = correction_json['message']
            self.corrections.append(correction)

    def is_reached(self, current_pose):
        for correction in self.corrections:
            paf1, confidence1 = get_paf(current_pose, correction['joint_1'])
            paf2, confidence2 = get_paf(current_pose, correction['joint_2'])
            if confidence1 == math.inf or confidence2 == math.inf:
                return False, correction['message']

            observed_angle = angle(paf1, paf2)
            angle_confidence = confidence1 + confidence2
            possible_deviation = observed_angle * angle_confidence
            if abs(abs(observed_angle - correction['angle']) - possible_deviation) > correction['deviation']:
                return False, correction['message']
        return True, self.message
