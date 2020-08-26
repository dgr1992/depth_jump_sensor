import traceback
import rospy
import tf
import numpy as np
import time
import math
import threading
import os

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from gap_visualiser import GapVisualiser

class GapSensor:

    def __init__(self):
        rospy.init_node("depth_jump_sensor")
        
        self.lock = threading.Lock()

        self.robot_yaw = 0
        self.robot_yaw_old = 0 
        self.odom_available = False

        self.robot_move_turning = False
        self.robot_move_forwards = False
        self.robot_move_backwards = False
        
        self.depth_jumps = []
        self.first = True

        self.min_depth_jump = 0.55 # 0.75
        self.max_r_to_depth_jump = 10 # 2.5

        self.scan = None
        self.scan_old = None
        self.scan_available = False
        
        self.shift_zero_count = 0
        self.shift_old = 0

        self.update_frequence = 90

        self.debug_to_file = True

        self._remove_debug_files()

        self._init_publishers()
        self._init_subscribers()

    def _init_publishers(self):
        """
        Initialise publishers
        """
        pass

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
        twist = Twist()
        twist = data
        if twist.angular.z != 0:
            self.robot_move_turning = True
        else:
            self.robot_move_turning = False

        if twist.linear.x != 0:
            if twist.linear.x > 0:
                self.robot_move_forwards = True
                self.robot_move_backwards = False
            else:
                self.robot_move_forwards = False
                self.robot_move_backwards = True
        else:
            self.robot_move_forwards = False
            self.robot_move_backwards = False

    def _receive_scan(self, data):
        """
        Receive laser scan data
        """
        self.scan_available = True
        if self.odom_available:
            self._process_range_data(data, self.robot_move_forwards, self.robot_move_backwards, self.robot_yaw)
        
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

        gap_visualisation_gnt = GapVisualiser('GNT')

        while not rospy.is_shutdown():

            if self.odom_available and self.scan_available and (time.time() - last_time) > (1.0/self.update_frequence):
                jumps = np.zeros(len(self.depth_jumps))
                for i in range(0, len(self.depth_jumps)):
                    if self.depth_jumps[i] >= 2:
                        jumps[i] = 1
                last_time = time.time()
                gap_visualisation_gnt.draw_gaps(jumps)

        gap_visualisation_gnt.close()
    
    def _process_range_data(self, scan, robot_move_forwards, robot_move_backwards, robot_yaw):
        """
        """
        self.lock.acquire()
        try:
            if self.first:
                self.scan = scan
                self.depth_jumps = np.zeros(len(scan.ranges))
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

                shift = self._calculate_shift(robot_yaw)

                # first: check for movement or appear in dj_2
                # second: dj_2 contains information where change happen. Use dj_1 to find nearest point and also for stand still
                self.depth_jumps = self._update(self.depth_jumps, dj_2, dj_1, shift, self.robot_move_forwards, self.robot_move_backwards)
                if self.debug_to_file:
                    with open('depth_jumps.csv','ab') as f4:
                        tmp = np.asarray(self.depth_jumps)
                        np.savetxt(f4, tmp.reshape(1, tmp.shape[0]), delimiter=",")
        except Exception as ex:
            print(ex)
            print(traceback.format_exc())
        
        self.lock.release()

    def _calculate_shift(self, robot_yaw):
        """
        Returns:
        shift (int): +1 -> rotation right; -1 -> rotation left
        """
        shift = self._calc_angle_change(robot_yaw)
        self.robot_yaw_old = robot_yaw

        if abs(shift) < 0.1:
            self.shift_zero_count += 1
            if self.shift_zero_count < 5:
                shift = self.shift_old
            else:
                shift = 0
        else:
            self.shift_zero_count = 0
            if shift < 0:
                shift = -1
            else:
                shift = +1
            self.shift_old = shift

        return shift

    def _calc_angle_change(self, robot_yaw):
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

    def _update(self, depth_jumps_last, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan, shift, robot_move_forwards, robot_move_backwards):
        # rotation
        if shift > 0:
            depth_jumps_last = self._correction_robot_rotation(depth_jumps_last, 0, len(depth_jumps_last), 1, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)
        elif shift < 0:      
            depth_jumps_last = self._correction_robot_rotation(depth_jumps_last, len(depth_jumps_last) - 1, -1, -1, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)
        
        # forwards backwards  
        if self.robot_move_forwards or self.robot_move_backwards:
            depth_jumps_last = self._correction_forward_backwards(depth_jumps_last, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan, robot_move_forwards, robot_move_backwards)

        return depth_jumps_last

    def _correction_robot_rotation(self, depth_jumps, start_index, end_index, increment, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan):
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
                
                if index_old != None:
                    # move
                    depth_jumps[index] = depth_jumps[index_old]
                    if depth_jumps[index] < 8:
                        depth_jumps[index] += 1
                    if index_old != index:
                        depth_jumps[index_old] = 0
                else:
                    # add
                    depth_jumps[index] = 1
            else:
                if depth_jumps[i] > 0:
                    # check depth jump from t - 1 is visible in single scan analysis
                    if depth_jumps_detected_from_single_scan[i] == 1:
                        if depth_jumps[i] < 8:
                            depth_jumps[i] += 1
                    # Depth jumps get removed over time. If at t -1 a depth jump was detected and at time t no depth jump then decrease.
                    else:
                        depth_jumps[i] -= 1
            
        return depth_jumps

    def _correction_drift_still_stand(self, array_detected_depth_jumps):
        for i in range(0, 360):
            if ((self.gnt_root.depth_jumps[i] != None and array_detected_depth_jumps[i] == 0)):                
                index_new = None
                increment = 0

                if array_detected_depth_jumps[i - 1] == 1:
                    index_new = i - 1
                    increment = -1
                elif array_detected_depth_jumps[(i + 1) % len(array_detected_depth_jumps)] == 1:
                    index_new = i + 1
                    increment = +1

                if index_new != None:
                    index = index_new % len(array_detected_depth_jumps)
                    
                    # if on new position already node then move this aswell
                    if self.gnt_root.depth_jumps[index] != None and self.gnt_root.depth_jumps[index].move_direction == self.gnt_root.depth_jumps[i].move_direction:
                        self._correction_robot_rotation(index_new, index_new + (abs(index_new - i) + 2) * increment, increment, array_detected_depth_jumps)

                    self._check_move_merge_disappear(i, index)

    def _correction_forward_backwards(self, array_detected_depth_jumps, robot_move_forwards, robot_move_backwards):
        if robot_move_forwards:
            self._correction_forward(array_detected_depth_jumps)
        if robot_move_backwards:
            self._correction_backwards(array_detected_depth_jumps)

    def _correction_forward(self, depth_jumps):
        # 0 -> 179
        for i in range(0, len(array_detected_depth_jumps) / 2):
            index_new = None

            # move, merge, disappear
            if (self.gnt_root.depth_jumps[i] != None and array_detected_depth_jumps[i] == 0):
                index_new = self._search_x_degree_positiv(array_detected_depth_jumps, i, 3)
                if index_new == None:
                    index_new = self._search_x_degree_negativ(array_detected_depth_jumps, i, 1)
                self._check_move_merge_disappear(i, index_new)

        # 359 -> 180
        for i in range(len(array_detected_depth_jumps) - 1, len(array_detected_depth_jumps) / 2, -1):
            index_new = None

            # move, merge, disappear
            if (self.gnt_root.depth_jumps[i] != None and array_detected_depth_jumps[i] == 0):
                index_new = self._search_x_degree_negativ(array_detected_depth_jumps, i, 3)
                if index_new == None:
                    index_new = self._search_x_degree_positiv(array_detected_depth_jumps, i, 1)
                self._check_move_merge_disappear(i, index_new)

    def _correction_backwards(self, array_detected_depth_jumps):
        # 180 -> 0
        for i in range(len(array_detected_depth_jumps) / 2, -1, -1):
            index_new = None
            # move, merge, disappear
            if (self.gnt_root.depth_jumps[i] != None and array_detected_depth_jumps[i] == 0):
                index_new = self._search_x_degree_negativ(array_detected_depth_jumps, i, 3)
                if index_new == None:
                    index_new = self._search_x_degree_positiv(array_detected_depth_jumps, i, 1)
                self._check_move_merge_disappear(i, index_new)

        # 180 -> 359
        for i in range(len(array_detected_depth_jumps) / 2, len(array_detected_depth_jumps)):
            index_new = None
            # move, merge, disappear
            if (self.gnt_root.depth_jumps[i] != None and array_detected_depth_jumps[i] == 0):
                index_new = self._search_x_degree_positiv(array_detected_depth_jumps, i, 3)
                if index_new == None:
                    index_new = self._search_x_degree_negativ(array_detected_depth_jumps, i, 1)
                self._check_move_merge_disappear(i, index_new)
    
    def _search_x_degree_positiv(self, array_detected_depth_jumps, index, degree_search):
        index_new = None

        for increment in range(1, degree_search + 1):
            if array_detected_depth_jumps[(index + increment)%len(self.gnt_root.depth_jumps)] == 1:
                index_new = (index + increment)%len(self.gnt_root.depth_jumps)
                break

        return index_new

    def _search_x_degree_negativ(self, array_detected_depth_jumps, index, degree_search):
        index_new = None

        for decrement in range(1, degree_search + 1):
            if array_detected_depth_jumps[index - decrement] == 1:
                index_new = index - decrement
                break

        return index_new

if __name__ == "__main__":
    try:
        gs = GapSensor()
        gs.run()
    except Exception as ex:
        print(ex.message)
