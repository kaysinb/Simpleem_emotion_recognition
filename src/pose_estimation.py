import numpy as np
import cv2


class PoseEstimation:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, img_size=(480, 640)):

        self.size = img_size
        self.model_points = np.array([[-32.97848, - 38.873894, - 1.4260463],  # pupil_right
                                      [32.63183, - 38.498325, - 1.1897595],  # pupil_left
                                      [1.226783, - 8.414541, 36.94806],     # nose
                                      [-28.916267, 28.612717, 2.24031],     # mouth left
                                      [28.794413, 28.079924, 3.217393]])    # mouth right

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])

    def solve_pose(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """

        (_, rotation_vector, translation_vector) = cv2.solvePnP(np.array(self.model_points), image_points,
                                                                self.camera_matrix, self.dist_coefs,
                                                                flags=cv2.SOLVEPNP_EPNP)

        return rotation_vector, translation_vector

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)
        cv2.circle(image, (point_2d[6][0], point_2d[6][1]), 3, (0, 0, 255), -1)
        cv2.circle(image, (point_2d[6][0], point_2d[6][1]), 3, (0, 0, 255), -1)

    @classmethod
    def rot_params_rv(cls, rvecs):

        from math import pi, atan2, asin
        R = cv2.Rodrigues(rvecs)[0]
        roll = 180 * atan2(-R[2][1], R[2][2]) / pi
        pitch = 180 * asin(R[2][0]) / pi
        yaw = 180 * atan2(-R[1][0], R[0][0]) / pi

        if abs(roll) > 90: roll = abs(180 - abs(roll)) * roll / abs(roll)
        if abs(pitch) > 90: pitch = abs(180 - abs(pitch)) * pitch / abs(pitch)
        if abs(yaw) > 90: yaw = abs(180 - abs(yaw)) * yaw / abs(yaw)

        rot_params = [roll, pitch, yaw]
        return rot_params


class Stabilizer:
    """Using Kalman filter as a point stabilizer."""

    def __init__(self,
                 state_num=4,
                 measure_num=2,
                 cov_process=0.1,
                 cov_measure=1):
        """Initialization"""
        # Currently we only support scalar and point, so check user input first.
        assert state_num == 4 or state_num == 2, "Only scalar and point supported, Check state_num please."

        # Store the parameters.
        self.state_num = state_num
        self.measure_num = measure_num

        # The filter itself.
        self.filter = cv2.KalmanFilter(state_num, measure_num, 0)

        # Store the state.
        self.state = np.zeros((state_num, 1), dtype=np.float32)

        # Store the measurement result.
        self.measurement = np.array((measure_num, 1), np.float32)

        # Store the prediction.
        self.prediction = np.zeros((state_num, 1), np.float32)

        # Kalman parameters setup for scalar.
        if self.measure_num == 1:
            self.filter.transitionMatrix = np.array([[1, 1],
                                                     [0, 1]], np.float32)

            self.filter.measurementMatrix = np.array([[1, 1]], np.float32)

            self.filter.processNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * cov_process

            self.filter.measurementNoiseCov = np.array(
                [[1]], np.float32) * cov_measure

        # Kalman parameters setup for point.
        if self.measure_num == 2:
            self.filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                     [0, 1, 0, 1],
                                                     [0, 0, 1, 0],
                                                     [0, 0, 0, 1]], np.float32)

            self.filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                      [0, 1, 0, 0]], np.float32)

            self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32) * cov_process

            self.filter.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], np.float32) * cov_measure

    def update(self, measurement):
        """Update the filter"""
        # Make kalman prediction
        self.prediction = self.filter.predict()

        # Get new measurement
        if self.measure_num == 1:
            self.measurement = np.array([[np.float32(measurement[0])]])
        else:
            self.measurement = np.array([[np.float32(measurement[0])],
                                         [np.float32(measurement[1])]])

        # Correct according to measurement
        self.filter.correct(self.measurement)

        # Update state value.
        self.state = self.filter.statePost

    def set_q_r(self, cov_process=0.1, cov_measure=0.001):
        """Set new value for processNoiseCov and measurementNoiseCov."""
        if self.measure_num == 1:
            self.filter.processNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array(
                [[1]], np.float32) * cov_measure
        else:
            self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], np.float32) * cov_measure
