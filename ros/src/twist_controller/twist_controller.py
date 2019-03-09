from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        self.wheel_base = kwargs['wheel_base']
        self.steer_ratio = kwargs['steer_ratio']
        self.max_lat_accel = kwargs['max_lat_accel']
        self.max_steer_angle = kwargs['max_steer_angle']

        self.vehicle_mass = kwargs['vehicle_mass']
        self.fuel_capacity = kwargs['fuel_capacity']
        self.brake_deadband = kwargs['brake_deadband']
        self.decel_limit = kwargs['decel_limit']
        self.accel_limit = kwargs['accel_limit']
        self.wheel_radius = kwargs['wheel_radius']
        self.pid_gain = kwargs['pid_gain']

        self.yaw_controller = YawController(self.wheel_base, self.steer_ratio, 0.1, self.max_lat_accel, self.max_steer_angle)

        self.throttle_controller = PID(self.pid_gain)

        self.LPF = LowPassFilter(1, 0.5)
        self.last_timestamp = rospy.get_time()

    def control(self, *args, **kwargs):
        dbw_enabled = kwargs['dbw_enabled']
        curr_linear_velocity = kwargs['curr_linear_velocity']
        cmd_linear_velocity = kwargs['cmd_linear_velocity']
        cmd_angular_velocity = kwargs['cmd_angular_velocity']

        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        curr_linear_velocity = self.LPF.filt(curr_linear_velocity)
        steering = self.yaw_controller.get_steering(cmd_linear_velocity, cmd_angular_velocity, curr_linear_velocity)
        vel_error = cmd_linear_velocity - curr_linear_velocity
        self.last_vel = curr_linear_velocity

        curr_timestamp = rospy.get_time()
        sample_time = curr_timestamp - self.last_timestamp
        self.last_timestamp = curr_timestamp

        # Set throttle using PID controller
        throttle = self.throttle_controller.step(vel_error, sample_time)

        # Set brake under different conditions
        brake = 0
        if cmd_linear_velocity == 0 and curr_linear_velocity < 0.1:
            throttle = 0
            brake = 500
        elif throttle < .1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steering
