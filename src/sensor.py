import traceback
import rospy
import tf
import numpy as np
import time
import math
import threading
import os

from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from depth_jump_sensor.msg import DepthJump

from gap_visualiser import GapVisualiser

class DepthJumpSensor:

    def __init__(self):
        rospy.init_node("depth_jump_sensor")
        
        self.lock = threading.Lock()

        self.robot_yaw = 0
        self.robot_yaw_old = 0
        self.rotation_zero_count = 0
        self.rotation_old = 0
        self.odom_available = False
        self.robot_move = 0
        
        self.depth_jumps = []
        self.depth_jumps_valid = []
        self.first = True

        self.min_depth_jump = 0.4 # 0.75
        self.max_r_to_depth_jump = 10 # 2.5

        self.max_depth_jump_recognition_count = 20
        self.depth_jump_recognition_threshold = 10
        self.recognition_decrease_rate = 1
        self.recognition_increase_rate = 1

        self.scan = None
        self.scan_old = None
        self.scan_available = False
        
        self.update_frequence = 90

        self.debug_to_file = False

        self.int32_max = 4294967295
        self.seq = 0

        self._remove_debug_files()

        self._init_publishers()
        self._init_subscribers()

    def _init_publishers(self):
        """
        Initialise publishers
        """
        self.pub_depth_jumps = rospy.Publisher('depth_jumps',DepthJump,queue_size=1)

    def _init_subscribers(self):
        """
        Initialise subscribers
        """
        rospy.Subscriber('scan', LaserScan, self._receive_scan)
        rospy.Subscriber('odom', Odometry, self._receive_odom)
        rospy.Subscriber('cmd_vel', Twist, self._receive_twist)

    def _receive_odom(self, data):
        """
        Receive odometrie data
        """
        # transform quarternion to euler to get robot yaw
        quaternion = (
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        roll = euler[0]
        pitch = euler[1]
        # convert to range 0 - 360
        self.robot_yaw = (((euler[2] + 2*math.pi) % (2*math.pi)) * 360)/(2*math.pi)

        if not self.odom_available:
            self.robot_yaw_old = self.robot_yaw

        self.odom_available = True

    def _receive_twist(self, data):
        """
        Receive twist information
        """
        twist = Twist()
        twist = data

        if twist.linear.x != 0:
            if twist.linear.x > 0:
                # forwards
                self.robot_move = 1
            else:
                # backwars
                self.robot_move = -1
        else:
            # stop
            self.robot_move = 0

    def _receive_scan(self, data):
        """
        Receive laser scan data
        """
        self.scan_available = True
        if self.odom_available:
            self._process(data, self.robot_move, self.robot_yaw)
        
    def _remove_debug_files(self):
        depth_jumps = "depth_jumps.csv"
        if os.path.exists(depth_jumps):
            os.remove(depth_jumps)

        depth_jumps = "dj_1.csv"
        if os.path.exists(depth_jumps):
            os.remove(depth_jumps)

        depth_jumps = "dj_2.csv"
        if os.path.exists(depth_jumps):
            os.remove(depth_jumps)

        depth_jumps = "scan.csv"
        if os.path.exists(depth_jumps):
            os.remove(depth_jumps)

    def run(self):
        angle_change = 0
        last_time = 0

        gap_visualisation_gnt = GapVisualiser('Depth Jump Sensor')

        while not rospy.is_shutdown():

            if self.odom_available and self.scan_available and (time.time() - last_time) > (1.0/self.update_frequence):
                last_time = time.time()
                gap_visualisation_gnt.draw_gaps(self.depth_jumps_valid)

        gap_visualisation_gnt.close()
    
    def _process(self, scan, robot_move, robot_yaw):
        """
        Starts the process of determing and updating the depth jumps using the given parameters.

        Parameters:
        scan (Scan): laser scan information to process
        rotation (int): rotation direction of robot for given scan
        robot_yaw (int): yaw of robot for given scan
        """
        self.lock.acquire()
        try:
            if self.first:
                self.scan = scan
                self.depth_jumps = np.zeros(len(scan.ranges))
                self.depth_jumps_valid = np.zeros(len(scan.ranges))
                self.first = False
            else:
                if self.debug_to_file:
                    with open('scan.csv','ab') as f1:
                        tmp = np.asarray(scan.ranges)
                        np.savetxt(f1, tmp.reshape(1, tmp.shape[0]), delimiter=",")

                self.scan_old = self.scan
                self.scan = scan

                dj_1 = self._find_depth_jumps_using_one_scan(self.scan.ranges)
                if self.debug_to_file:
                    with open('dj_1.csv','ab') as f2:
                        tmp = np.asarray(dj_1)
                        np.savetxt(f2, tmp.reshape(1, tmp.shape[0]), delimiter=",")

                dj_2 = self._find_depth_jumps_using_two_scans(self.scan.ranges, self.scan_old.ranges)
                if self.debug_to_file:
                    with open('dj_2.csv','ab') as f3:
                        tmp = np.asarray(dj_2)
                        np.savetxt(f3, tmp.reshape(1, tmp.shape[0]), delimiter=",")

                rotation = self._get_rotation_direction(robot_yaw)

                self.depth_jumps = self._update(self.depth_jumps, dj_2, dj_1, rotation, robot_move)
                if self.debug_to_file:
                    with open('depth_jumps.csv','ab') as f4:
                        tmp = np.asarray(self.depth_jumps)
                        np.savetxt(f4, tmp.reshape(1, tmp.shape[0]), delimiter=",")

                self.depth_jumps_valid = self._get_valid_depth_jumps(self.depth_jump_recognition_threshold)

                # publish
                self._publish_data(self.depth_jumps_valid, self.scan.ranges, rotation, robot_move)

        except Exception as ex:
            print(ex)
            print(traceback.format_exc())
        
        self.lock.release()

    def _get_valid_depth_jumps(self, min_times_detected):
        """
        From the array of depth jumps use all depth jums that were detected x times.

        Parameters:
        min_times_detected (int): Threshold for detection that determines valid
        """
        jumps = np.zeros(len(self.depth_jumps))
        for i in range(0, len(self.depth_jumps)):
            if self.depth_jumps[i] >= min_times_detected:
                jumps[i] = 1
        return jumps

    def _get_rotation_direction(self, robot_yaw):
        """
        Get the current rotation direction of the robot.
        
        Parameters:
        robot_yaw (float): yaw of the robot in degree

        Returns:
        rotation (int): +1 -> left; -1 -> right
        """
        rotation = self._calc_angle_change(robot_yaw)
        self.robot_yaw_old = robot_yaw

        if abs(rotation) < 0.1:
            self.rotation_zero_count += 1
            if self.rotation_zero_count < 5:
                rotation = self.rotation_old
            else:
                rotation = 0
        else:
            self.rotation_zero_count = 0
            if rotation < 0:
                rotation = -1
            else:
                rotation = +1
            self.rotation_old = rotation

        return rotation

    def _calc_angle_change(self, robot_yaw):
        """
        Calculate the angle change between last angle and current angel.

        Parameters:
        robot_yaw (float): yaw of the robot in degree

        Returns:
        angle_change (float): angel change in degree
        """
        angle_change = 0
        # detect jump 359 <-> 0
        if robot_yaw > 350 and self.robot_yaw_old < 10:
            # angle decreased
            angle_change = (-1) * (360.0 - robot_yaw + self.robot_yaw_old)
        elif robot_yaw < 5 and self.robot_yaw_old > 350:
            # angle increased
            angle_change = 360.0 - self.robot_yaw_old + robot_yaw
        else:
            angle_change = float(robot_yaw) - self.robot_yaw_old

        return angle_change

    def _find_depth_jumps_using_two_scans(self, scan, scan_old):
        """
        Calculate depth jumps using scan_{t-1} and scan_{t}.

        Parameters:
        scan (float[]): range data from time t
        scan_old (float[]): range data from time t-1

        Returns:
        depth_jumps (int[]): A 1 indicates a depth jump
        """

        scan = np.asarray(scan)
        scan_old = np.asarray(scan_old)
        scan[scan == np.inf] = self.max_r_to_depth_jump
        scan_old[scan_old == np.inf] = self.max_r_to_depth_jump

        # works for rotation but fails for moving forwards or backwards
        depth_jumps = abs(scan - scan_old)
        depth_jumps[depth_jumps < self.min_depth_jump] = 0
        depth_jumps[depth_jumps > 0] = 1
        depth_jumps = depth_jumps.astype(int)

        return depth_jumps

    def _find_depth_jumps_using_one_scan(self, scan):
        """
        Caclulate depth jumps using only scan_{t}.

        Parameters:
        scan (float[]): range data from time t

        Returns:
        depth_jumps (int[]): A 1 indicates a depth jump
        """

        # works for rotation, forwards and backwards but detects two points on the same wall as a gap if they are far enough apart
        depth_jumps = np.zeros((len(scan),), dtype=int)
        for angle in range(0, len(scan)):
            tmp_angle = -2
            # check if jump is large enough
            if (scan[angle] != np.inf and scan[(angle + 1) % 360] != np.inf and abs(scan[angle] - scan[(angle + 1) % 360]) >= self.min_depth_jump):
                # find the shortest r
                tmp_angle = angle

                if scan[angle] > scan[(angle + 1) % 360]:
                    tmp_angle = (angle + 1) % 360
            
            elif scan[angle] == np.inf and scan[(angle + 1) % 360] != np.inf:
                tmp_angle = (angle + 1) % 360

            elif scan[angle] != np.inf and scan[(angle + 1) % 360] == np.inf:
                tmp_angle = angle

            if tmp_angle != -2 and scan[tmp_angle] < self.max_r_to_depth_jump:
                # shortest r must be smaller then max distance
                depth_jumps[tmp_angle] = 1
        
        return depth_jumps

    def _update(self, depth_jumps_last, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan, rotation, robot_move):
        """
        Update the depth jumps from t - 1 to t by checking the rotation and movement

        Parameters:
        depth_jumps_last (int[]): depth jumps at t - 1 
        depth_jumps_detected_from_two_scans (int[]): depth jumps calculated from scan_{t} and scan_{t - 1} 
        depth_jumps_detected_from_single_scan (int[]): depth jumps calculated from scan_{t}
        rotation (int): rotation direction of robot
        robot_move (int): movement direction (forward / backwords)

        Returns:
        depth_jumps_last (int[]): updated depth jumps
        """

        # rotation
        if rotation > 0:
            depth_jumps_last = self._correction(depth_jumps_last, 0, len(depth_jumps_last), 1, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)
        elif rotation < 0:      
            depth_jumps_last = self._correction(depth_jumps_last, len(depth_jumps_last) - 1, -1, -1, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)
        
        # forwards backwards movement
        if robot_move > 0:
            depth_jumps_last = self._correction_forward(depth_jumps_last, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)
        elif robot_move < 0:
            depth_jumps_last = self._correction_backwards(depth_jumps_last, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)

        return depth_jumps_last

    def _correction(self, depth_jumps, start_index, end_index, increment, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan):
        """
        Corrects the last state of the depth_jumps to the current for the given angle.

        Parameters:
        depth_jumps_last (int[]): depth jumps at t - 1 
        start_index (int): angle at which correction shall start
        end_index (int): angle at which correction shall stop
        increment (int): the increment to go towards the end index
        depth_jumps_detected_from_two_scans (int[]): depth jumps calculated from scan_{t} and scan_{t - 1} 
        depth_jumps_detected_from_single_scan (int[]): depth jumps calculated from scan_{t}

        Returns:
        depth_jumps_last (int[]): updated depth jumps
        """

        for i in range(start_index, end_index, increment):
            # detected depth jump using scan_{t-1} - scan_{t}
            if depth_jumps_detected_from_two_scans[i % len(depth_jumps_detected_from_two_scans)] == 1:
                index = i
                index_old = None

                # check if marking is at closest point using depth_jumps_single_scan
                if depth_jumps_detected_from_single_scan[index] == 0:
                    if depth_jumps_detected_from_single_scan[(index + increment) % len(depth_jumps_detected_from_single_scan)] == 1:
                        index = (index + increment) % len(depth_jumps_detected_from_single_scan)
                    elif depth_jumps_detected_from_single_scan[(index - increment) % len(depth_jumps_detected_from_single_scan)] == 1:
                        index = (index - increment) % len(depth_jumps_detected_from_single_scan)

                # find corresponding position of depth jump at t-1
                for j in range(0, 3):
                    if depth_jumps[(index + increment * j) % len(depth_jumps)] > 0:
                        index_old = (index + increment * j) % len(depth_jumps)
                        break
                
                # corresponding position might be in the opposite direction
                if index_old == None and depth_jumps[(index - increment) % len(depth_jumps)] > 0:
                    index_old = (index - increment) % len(depth_jumps)

                if index_old != None:
                    # move
                    depth_jumps[index] = depth_jumps[index_old]
                    if depth_jumps[index] < self.max_depth_jump_recognition_count:
                        depth_jumps[index] += self.recognition_increase_rate
                    if index_old != index:
                        depth_jumps[index_old] = 0
                else:
                    # add
                    depth_jumps[index] = 1
            else:
                if depth_jumps[i] > 0:
                    # check depth jump from t - 1 is visible in single scan analysis
                    if depth_jumps_detected_from_single_scan[i] == 1:
                        if depth_jumps[i] < self.max_depth_jump_recognition_count:
                            depth_jumps[i] += self.recognition_increase_rate
                    # Depth jumps get removed over time. If at t -1 a depth jump was detected and at time t no depth jump then decrease.
                    else:
                        depth_jumps[i] -= self.recognition_decrease_rate
                else:
                    depth_jumps[i] = 0
            
        return depth_jumps     

    def _correction_forward(self, depth_jumps_last, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan):
        """
        Correction of backwards movement.

        Parameters: 
        depth_jumps_last (int[]): depth jumps at t - 1 
        depth_jumps_detected_from_two_scans (int[]): depth jumps calculated from scan_{t} and scan_{t - 1} 
        depth_jumps_detected_from_single_scan (int[]): depth jumps calculated from scan_{t}

        Return:
        depth_jumps_last (int[]): updated depth jumps
        """

        # 0 -> 179
        start = 0
        end = len(depth_jumps_last) / 2
        increment = 1
        depth_jumps_last = self._correction(depth_jumps_last, start, end, increment, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)

        # 359 -> 180
        start = len(depth_jumps_last) - 1
        end = len(depth_jumps_last) / 2
        increment = -1
        depth_jumps_last = self._correction(depth_jumps_last, start, end, increment, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)

        return depth_jumps_last

    def _correction_backwards(self, depth_jumps_last, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan):
        """
        Correction of backwards movement.

        Parameters: 
        depth_jumps_last (int[]): depth jumps at t - 1 
        depth_jumps_detected_from_two_scans (int[]): depth jumps calculated from scan_{t} and scan_{t - 1} 
        depth_jumps_detected_from_single_scan (int[]): depth jumps calculated from scan_{t}

        Return:
        depth_jumps_last (int[]): updated depth jumps
        """
        # 180 -> 0
        start = len(depth_jumps_last) / 2
        end = -1
        increment = -1
        depth_jumps_last = self._correction(depth_jumps_last, start, end, increment, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)

        # 180 -> 359
        start = len(depth_jumps_last) / 2
        end = len(depth_jumps_last)
        increment = 1
        depth_jumps_last = self._correction(depth_jumps_last, start, end, increment, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)

        return depth_jumps_last

    def _publish_data(self, depth_jumps, range_data, rotation, movement):
        """
        Publish depth jump information.

        Parameters:
        depth_jumps (int[]): 0 = no depth jump , 1 = depth jump
        range_data (floag[]): the according range data to the depth jump result
        rotation (int): rotation direction of robot, 1 = left, -1 = right
        movement (int): movement of robot; 1 = forward, -1 = backwards
        """
        if self.seq == self.int32_max:
            self.seq = 0
        self.seq += 1

        djmsg = DepthJump()
        djmsg.header.stamp = rospy.Time.now()
        djmsg.header.seq = self.seq 
        djmsg.depth_jumps = depth_jumps
        djmsg.range_data = range_data
        djmsg.rotation = rotation
        djmsg.liniear_x = movement
        self.pub_depth_jumps.publish(djmsg)

if __name__ == "__main__":
    try:
        gs = DepthJumpSensor()
        gs.run()
    except Exception as ex:
        print(ex.message)
