#!/usr/bin/env python3

################################################################################

# Copyright (c) 2026, Tinker Twins
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

################################################################################

# ROS 2 module imports
import rclpy # ROS 2 client library (rcl) for Python (built on rcl C API)
from rclpy.node import Node # Base class for ROS 2 nodes
from rclpy.qos import QoSProfile # Ouality of Service (tune communication between nodes)
from rclpy.signals import SignalHandlerOptions # Control ROS 2 signal handling
from std_msgs.msg import Float32 # Float32 message class
from sensor_msgs.msg import LaserScan # LaserScan message class
from nav_msgs.msg import Odometry # Odometry message class

# Python module imports
import math # Mathematical functions

################################################################################

# Constrain control commands
def constrain(input, low_bound, high_bound):
    return max(low_bound, min(high_bound, input))

# Rectified linear unit
def relu(x):
    return max(0.0, x)

################################################################################

class WallFollowNode(Node):
    def __init__(self):
        super().__init__('wall_follow_node')

        # Parameters (tunable via --ros-args -p name:=value)
        self.desired_distance = self.declare_parameter('desired_distance', 0.8).value # Distance to keep from a lone wall until track width is measured [m]
        self.kp = self.declare_parameter('kp', 1.0).value # Proportional gain on offset from track center
        self.kd = self.declare_parameter('kd', 0.1).value # Derivative gain
        self.ki_steer = self.declare_parameter('ki_steer', 0.05).value # Integral gain on offset from track center
        self.i_max = self.declare_parameter('i_max', 1.0).value # Anti-windup clamp on steering integral [m*s]
        self.lookahead = self.declare_parameter('lookahead', 0.5).value # Lookahead distance [m]
        self.wall_max_range = self.declare_parameter('wall_max_range', 2.5).value # Beyond this a wall is treated as missing (opening) [m]
        self.theta = math.radians(50.0) # Angle between the two wall beams

        # Gap steering parameters (steer toward the most open direction ahead to turn into corners early)
        self.k_gap = self.declare_parameter('k_gap', 0.3).value # Steering per radian of open-direction angle
        self.gap_fov = math.radians(self.declare_parameter('gap_fov_deg', 70.0).value) # Half-width of search window [rad]

        # Corner detection parameters (forward cone min distance flags a corner, side cones pick its direction)
        self.d_corner = self.declare_parameter('d_corner', 1.5).value # Forward clearance below which a corner is detected [m]
        self.k_corner = self.declare_parameter('k_corner', 0.7).value # Steering per meter inside d_corner
        self.side_cone_center = math.radians(self.declare_parameter('side_cone_center_deg', 45.0).value) # Diagonal cone direction [rad]
        self.side_cone_width = math.radians(self.declare_parameter('side_cone_width_deg', 15.0).value) # Diagonal cone half-width [rad]

        # Speed planning parameters: v = v_max - k_front*relu(d_brake - d_front) - k_steer*relu(|steer| - steer_deadband)
        self.cone_angle = math.radians(self.declare_parameter('cone_angle_deg', 8.0).value) # Half-width of forward cone [rad]
        self.v_max = self.declare_parameter('v_max', 2.5).value # Target speed on straights [m/s]
        self.v_min = self.declare_parameter('v_min', 1.0).value # Minimum target speed in corners [m/s]
        self.d_brake = self.declare_parameter('d_brake', 3.0).value # Forward clearance at which slowing starts [m]
        self.k_front = self.declare_parameter('k_front', 0.6).value # Speed reduction per meter inside d_brake [(m/s)/m]
        self.k_steer = self.declare_parameter('k_steer', 2.0).value # Speed reduction per unit steering beyond deadband [m/s]
        self.steer_deadband = self.declare_parameter('steer_deadband', 0.3).value # Steering magnitude ignored by speed planner

        # Speed controller parameters (PI on measured speed -> throttle command)
        self.kv = self.declare_parameter('kv', 0.5).value # Proportional gain [throttle/(m/s)]
        self.ki = self.declare_parameter('ki', 0.2).value # Integral gain [throttle/(m*s^-1*s)]
        self.throttle_min = self.declare_parameter('throttle_min', 0.0).value # Lowest throttle (negative allows braking/reverse)
        self.throttle_max = self.declare_parameter('throttle_max', 1.0).value # Highest throttle

        # ROS 2 infrastructure
        qos = QoSProfile(depth=1)
        self.pub_steering_command = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', qos)
        self.pub_throttle_command = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', qos)
        self.sub_lidar = self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.scan_callback, qos)
        self.sub_odom = self.create_subscription(Odometry, '/autodrive/roboracer_1/odom', self.odom_callback, qos)

        # Initialize
        self.prev_error = 0.0
        self.prev_time = None
        self.prev_walls = None # Which walls were visible last scan
        self.steer_integral = 0.0 # Integral of offset from track center [m*s]
        self.track_width = None # Low-pass estimate of track width [m]
        self.speed = 0.0 # Measured speed [m/s]
        self.speed_integral = 0.0 # Integral of speed error [m]

        self.get_logger().info('Wall follow node started')

    # Get range at a given angle (0 rad = straight ahead, positive = left)
    def get_range(self, scan, angle):
        index = int(round((angle - scan.angle_min) / scan.angle_increment))
        index = constrain(index, 0, len(scan.ranges) - 1)
        r = scan.ranges[index]
        if math.isnan(r) or math.isinf(r):
            return scan.range_max
        return r

    # Get minimum valid range within +/- half_width of a center angle
    def get_cone_min(self, scan, center, half_width):
        start = int(round((center - half_width - scan.angle_min) / scan.angle_increment))
        end = int(round((center + half_width - scan.angle_min) / scan.angle_increment))
        start = constrain(start, 0, len(scan.ranges) - 1)
        end = constrain(end, 0, len(scan.ranges) - 1)
        valid = [r for r in scan.ranges[start:end + 1] if math.isfinite(r) and r >= scan.range_min]
        return min(valid) if valid else scan.range_max

    # Projected distance to the left (side=+1) or right (side=-1) wall, or None if that wall is missing
    def get_wall_distance(self, scan, side):
        b = self.get_range(scan, side * math.pi / 2) # Perpendicular beam
        a = self.get_range(scan, side * (math.pi / 2 - self.theta)) # Beam theta ahead of b
        if b > self.wall_max_range or a > self.wall_max_range / math.cos(self.theta):
            return None
        alpha = math.atan2(a * math.cos(self.theta) - b, a * math.sin(self.theta)) # Wall orientation
        return b * math.cos(alpha) + self.lookahead * math.sin(alpha)

    # Angle of the center of the widest most-open region ahead (0 rad = straight, positive = left)
    def get_gap_angle(self, scan):
        start = int(round((-self.gap_fov - scan.angle_min) / scan.angle_increment))
        end = int(round((self.gap_fov - scan.angle_min) / scan.angle_increment))
        start = constrain(start, 0, len(scan.ranges) - 1)
        end = constrain(end, 0, len(scan.ranges) - 1)
        ranges = [r if math.isfinite(r) else scan.range_max for r in scan.ranges[start:end + 1]]
        # Moving average to ignore single-beam spikes
        window = 9
        smoothed = [sum(ranges[max(0, i - window // 2):i + window // 2 + 1]) / len(ranges[max(0, i - window // 2):i + window // 2 + 1])
                    for i in range(len(ranges))]
        # Longest run of beams within 90% of the deepest reading
        threshold = 0.9 * max(smoothed)
        best_start, best_len, run_start = 0, 0, None
        for i, r in enumerate(smoothed + [-1.0]):
            if r >= threshold and run_start is None:
                run_start = i
            elif r < threshold and run_start is not None:
                if i - run_start > best_len:
                    best_start, best_len = run_start, i - run_start
                run_start = None
        center = start + best_start + best_len // 2
        return scan.angle_min + center * scan.angle_increment

    # Store measured speed (magnitude of planar velocity, frame independent)
    def odom_callback(self, odom):
        v = odom.twist.twist.linear
        self.speed = math.hypot(v.x, v.y)

    # Compute and publish control commands from each LiDAR scan
    def scan_callback(self, scan):
        if len(scan.ranges) == 0:
            return

        # Projected distances to both walls (None = opening on that side)
        d_left = self.get_wall_distance(scan, 1)
        d_right = self.get_wall_distance(scan, -1)
        half_width = self.track_width / 2 if self.track_width is not None else self.desired_distance

        # Offset from track center (positive = right of center = steer left)
        if d_left is not None and d_right is not None:
            error = (d_left - d_right) / 2
            width = d_left + d_right
            self.track_width = width if self.track_width is None else 0.9 * self.track_width + 0.1 * width
        elif d_left is not None:
            error = d_left - half_width
        elif d_right is not None:
            error = half_width - d_right
        else:
            error = 0.0 # No walls visible: rely on gap steering

        # PID control (skip derivative when visible walls change, since the error jumps)
        walls = (d_left is not None, d_right is not None)
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = (now - self.prev_time) if self.prev_time is not None else 0.0
        derivative = (error - self.prev_error) / dt if dt > 0.0 and walls == self.prev_walls else 0.0
        self.prev_error = error
        self.prev_time = now
        self.prev_walls = walls

        # Corner detection: wall closing in ahead -> turn toward the more open diagonal
        d_front = self.get_cone_min(scan, 0.0, self.cone_angle)
        corner_strength = relu(self.d_corner - d_front)
        in_corner = corner_strength > 0.0
        corner_steering = 0.0
        if in_corner:
            d_diag_left = self.get_cone_min(scan, self.side_cone_center, self.side_cone_width)
            d_diag_right = self.get_cone_min(scan, -self.side_cone_center, self.side_cone_width)
            corner_steering = self.k_corner * corner_strength * (1.0 if d_diag_left > d_diag_right else -1.0)

        # Add steering toward open space ahead so corners are entered early
        gap_angle = self.get_gap_angle(scan)
        steering_raw = (self.kp * error + self.ki_steer * self.steer_integral + self.kd * derivative
                        + self.k_gap * gap_angle + corner_steering)
        steering = constrain(steering_raw, -1.0, 1.0)

        # Integrate centering error only on unsaturated straights (corners and openings would wind it up)
        if in_corner or walls != (True, True):
            self.steer_integral *= 0.9 # Bleed off so the bias doesn't fight the turn
        elif steering == steering_raw:
            self.steer_integral = constrain(self.steer_integral + error * dt, -self.i_max, self.i_max)

        # Target speed: full speed on straights, slow down as forward clearance shrinks or steering grows
        v_target = (self.v_max
                    - self.k_front * relu(self.d_brake - d_front)
                    - self.k_steer * relu(abs(steering) - self.steer_deadband))
        v_target = constrain(v_target, self.v_min, self.v_max)

        # PI speed control (integrator only accumulates while throttle is unsaturated)
        speed_error = v_target - self.speed
        throttle_raw = self.kv * speed_error + self.ki * self.speed_integral
        throttle = constrain(throttle_raw, self.throttle_min, self.throttle_max)
        if throttle == throttle_raw:
            self.speed_integral += speed_error * dt

        self.publish_commands(throttle, steering)

    # Publish control messages
    def publish_commands(self, throttle, steering):
        throttle_msg = Float32()
        steering_msg = Float32()
        throttle_msg.data = float(throttle)
        steering_msg.data = float(steering)
        self.pub_throttle_command.publish(throttle_msg)
        self.pub_steering_command.publish(steering_msg)

    # Generate and publish zero commands
    def stop(self):
        self.publish_commands(0.0, 0.0)

################################################################################

def main():
    # Handle CTRL+C ourselves so the context is still valid to publish zero commands
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = WallFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

################################################################################

if __name__ == '__main__':
    main()
