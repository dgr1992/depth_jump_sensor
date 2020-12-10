import traceback
import rospy
import tf
import numpy as np
import time
import math
import threading
import os
import Queue

from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from depth_jump_sensor.msg import DepthJump

from gap_visualiser import GapVisualiser

class DepthJumpSensor:

    def __init__(self):
        rospy.init_node("depth_jump_sensor", log_level=rospy.INFO)
        
        self.lock = threading.Lock()

        self.robot_yaw = 0
        self.robot_yaw_old = 0
        self.rotation_zero_count = 0
        self.rotation_old = 0
        self.odom_available = False
        self.robot_move = 0
        self.rotation = 0
        
        self.depth_jumps = []
        self.discontinuities = []
        self.depth_discontinuities_single_scan = []
        self.first = True

        self.min_depth_jump = 0.4 # 0.75
        #self.max_r_to_depth_jump = 10 # 2.5
        self.max_r = 0

        self.max_depth_jump_recognition_count = 30
        self.depth_jump_recognition_threshold = 10
        self.recognition_decrease_rate = 1
        self.recognition_increase_rate = 1

        self.scan = None
        self.scan_old = None
        self.scan_available = False
        
        self.update_frequence = 90
        self.sum_processing_time = 0

        self.debug_to_file = False

        self.seq = 1

        self.queue = []

        self.scan_history = []
        self.dj_1_history = []
        self.dj_2_history = []
        self.depth_jumps_history = []

        self._remove_debug_files()

        self._init_publishers()
        self._init_subscribers()

    def _init_publishers(self):
        """
        Initialise publishers
        """
        self.pub_depth_jumps = rospy.Publisher('depth_jumps',DepthJump,queue_size=15)

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
        yaw = euler[2]
        # convert to range 0 degree to 360 degree
        self.robot_yaw = (((yaw + 2*math.pi) % (2*math.pi)) * 360)/(2*math.pi)

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

        #if twist.angular.z != 0:
        #    if twist.angular.z > 0:
        #        # left
        #        self.rotation = 1
        #    else:
        #        # right
        #        self.rotation = -1
        #else:
        #    # stop
        #    self.rotation = 0

    def _receive_scan(self, data):
        """
        Receive laser scan data
        """
        self.max_r = data.range_max
        self.scan_available = True
        if self.odom_available:
            self.queue.append((data, self.robot_move, self.robot_yaw))#, self.rotation))
            #self._process(data, self.robot_move, self.robot_yaw, self.rotation)
        
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
        #gap_visualisation_single_scan = GapVisualiser('Depth Discontinuities Single Scan')

        while not rospy.is_shutdown():
            if self.odom_available and self.scan_available:
            
                if (time.time() - last_time) > (1.0/self.update_frequence):
                    last_time = time.time()
                    gap_visualisation_gnt.draw_gaps(self.discontinuities)
                    #gap_visualisation_single_scan.draw_gaps(self.depth_discontinuities_single_scan)

                if self.queue:
                    scan, movement, robot_yaw = self.queue.pop(0)
                    self._process(scan, movement, robot_yaw)

        gap_visualisation_gnt.close()
        #gap_visualisation_single_scan.close()
        
        avg_processing = (self.sum_processing_time) / self.seq
        print("Time avg process time: " + str(avg_processing) + " ms")

        # save debug information
        self._save_scan_history()
        self._save_dj_1_history()
        self._save_dj_2_history()
        self._save_depth_jumps_history()
    
    def _process(self, scan, movement, robot_yaw):
        """
        Starts the process of determing and updating the depth jumps using the given parameters. 
        
        Parameters: 
        scan (Scan): laser scan information to process 
        rotation (int): rotation direction of robot for given scan 
        robot_yaw (int): yaw of robot for given scan 
        """
        self.lock.acquire()
        try:
            start = time.time()
            if self.first:
                self.scan = scan
                self.depth_jumps = np.zeros(len(scan.ranges))
                self.discontinuities = np.zeros(len(scan.ranges))
                self.first = False
            else:
                rotation = self._get_rotation_direction(robot_yaw)

                start_file_write = time.time()
                if self.debug_to_file:
                    with open('scan.csv','ab') as f1:
                        tmp = np.asarray(scan.ranges)
                        tmp = np.insert(tmp,0,rotation)
                        tmp = np.insert(tmp,0,movement)
                        tmp = np.insert(tmp,0,self.seq)
                        self.scan_history.append(tmp)
                
                self.scan_old = self.scan
                self.scan = scan

                dj_1 = self._find_depth_jumps_using_one_scan(self.scan.ranges)
                self.depth_discontinuities_single_scan = dj_1
                if self.debug_to_file:
                    with open('dj_1.csv','ab') as f2:
                        tmp = np.asarray(dj_1)
                        tmp = np.insert(tmp,0,rotation)
                        tmp = np.insert(tmp,0,movement)
                        tmp = np.insert(tmp,0,self.seq)
                        self.dj_1_history.append(tmp)

                dj_2 = self._find_depth_jumps_using_two_scans(self.scan.ranges, self.scan_old.ranges)
                if self.debug_to_file:
                    with open('dj_2.csv','ab') as f3:
                        tmp = np.asarray(dj_2)
                        tmp = np.insert(tmp,0,rotation)
                        tmp = np.insert(tmp,0,movement)
                        tmp = np.insert(tmp,0,self.seq)
                        self.dj_2_history.append(tmp)

                self.depth_jumps = self._update(self.depth_jumps, dj_2, dj_1, rotation, movement)
                if self.debug_to_file:
                    with open('depth_jumps.csv','ab') as f4:
                        tmp = np.asarray(self.depth_jumps)
                        tmp = np.insert(tmp,0,rotation)
                        tmp = np.insert(tmp,0,movement)
                        tmp = np.insert(tmp,0,self.seq)
                        self.depth_jumps_history.append(tmp)

                self.discontinuities = self._get_valid_depth_jumps()

                # publish
                self._publish_data(self.discontinuities, self.scan.ranges, rotation, movement)
               
                end = time.time()
                process_time = (end - start) * 1000

                self.sum_processing_time += process_time
        except Exception as ex:
            print(ex)
            print(traceback.format_exc())
        
        self.lock.release()

    def _get_valid_depth_jumps(self):
        """
        From the array of depth jumps use all depth jums that were detected x times. 
        
        Parameters: 
        min_times_detected (int): Threshold for detection that determines valid 
        
        Returns: 
        discontinuities (int[]): Array with same lenght as depth_jumps and discontinuities marked with 1 
        """
        discontinuities = np.zeros(len(self.depth_jumps))
        for i in range(0, len(self.depth_jumps)):
            if self.depth_jumps[i] >= self.depth_jump_recognition_threshold:
                discontinuities[i] = 1
        return discontinuities

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
        scan[scan == np.inf] = self.max_r + self.min_depth_jump
        scan_old[scan_old == np.inf] = self.max_r + self.min_depth_jump

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
        depth_jumps = np.zeros((len(scan),), dtype=int)
        for angle in range(0, len(scan)):
            tmp_angle = -2
            # check if jump is large enough
            if (scan[angle] != np.inf and scan[(angle + 1) % 360] != np.inf and abs(scan[angle] - scan[(angle + 1) % 360]) >= self.min_depth_jump):
                # find the shortest r
                tmp_angle = angle

                if scan[angle] > scan[(angle + 1) % 360]:
                    tmp_angle = (angle + 1) % 360

                depth_jumps[tmp_angle] = 1
            
            # if the measured distance is infinite at one of the angles select the angle where not infinite
            elif scan[angle] == np.inf and scan[(angle + 1) % 360] != np.inf:
                #tmp_angle = (angle + 1) % 360
                depth_jumps[(angle + 1) % 360] = 1
            elif scan[angle] != np.inf and scan[(angle + 1) % 360] == np.inf:
                #tmp_angle = angle
                depth_jumps[angle] = 1
        
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
        if rotation != 0:
            start = None
            end = None
            for_increment = None
            search_increment = None

            if rotation > 0:
                # rotation right
                start = 0
                end = len(depth_jumps_last)
                for_increment = 1
                search_increment = 1
            elif rotation < 0: 
                # rotation left
                start = len(depth_jumps_last) - 1
                end = -1
                for_increment = -1
                search_increment = -1
                 
            depth_jumps_last = self._correction(depth_jumps_last, start, end, for_increment, search_increment, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)
        
        # forwards backwards movement
        if robot_move > 0:
            depth_jumps_last = self._correction_forward(depth_jumps_last, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)
        elif robot_move < 0:
            depth_jumps_last = self._correction_backwards(depth_jumps_last, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)

        if robot_move == 0 and rotation == 0:
            start = 0
            end = len(depth_jumps_last)
            for_increment = 1
            search_increment = 1
            depth_jumps_last = self._correction(depth_jumps_last, start, end, for_increment, search_increment, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)


        return depth_jumps_last

    def _correction(self, depth_jumps, start_index, end_index, for_increment, search_increment, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan):
        """
        Corrects the last state of the depth_jumps to the current for the given angle. 

        Parameters: 
        depth_jumps (int[]): depth jumps at t - 1 
        start_index (int): angle at which correction shall start 
        end_index (int): angle at which correction shall stop 
        incremfor_incrementent (int): the for_increment to go towards the end index 
        search_increment (int): the increment to use finding the previous position of the depth jump 
        depth_jumps_detected_from_two_scans (int[]): depth jumps calculated from scan_{t} and scan_{t - 1} 
        depth_jumps_detected_from_single_scan (int[]): depth jumps calculated from scan_{t} 

        Returns: 
        depth_jumps_last (int[]): updated depth jumps 
        """
        for i in range(start_index, end_index, for_increment):
            # detected depth jump using scan_{t-1} - scan_{t}
            if depth_jumps_detected_from_two_scans[i % len(depth_jumps_detected_from_two_scans)] == 1:
                index = i
                index_old_1 = None
                index_old_2 = None
                split_detect = False

                rospy.logdebug("move - seq:" + str(self.seq) + " detect dj2 index=" + str(index))
                # check if marking is at closest point using depth_jumps_single_scan
                if depth_jumps_detected_from_single_scan[index] == 0:
                    if depth_jumps_detected_from_single_scan[(index + search_increment) % len(depth_jumps_detected_from_single_scan)] == 1:
                        index = (index + search_increment) % len(depth_jumps_detected_from_single_scan)
                    elif depth_jumps_detected_from_single_scan[(index - search_increment) % len(depth_jumps_detected_from_single_scan)] == 1:
                        index = (index - search_increment) % len(depth_jumps_detected_from_single_scan)
                
                rospy.logdebug("move - seq:" + str(self.seq) + " closest point index=" + str(index))

                # corresponding position of depth jump at t-1
                for j in range(0, 4):
                    if index_old_1 == None and depth_jumps[(index + search_increment * j) % len(depth_jumps)] > 0:
                        index_old_1 = (index + search_increment * j) % len(depth_jumps)
                    if index_old_2 == None and depth_jumps[(index - search_increment * j) % len(depth_jumps)] > 0:
                        index_old_2 = (index - search_increment * j) % len(depth_jumps)

                if index_old_1 != None or index_old_2 != None:
                    # check for split
                    split_detect, index_2 = self._check_for_split(depth_jumps, index, depth_jumps_detected_from_single_scan, index_old_1, index_old_2)
                    
                    #check for merge
                    merge_detect = self._check_for_merge(index_old_1, index_old_2, index_2)

                    if merge_detect and not split_detect:
                        rospy.logdebug("merge - seq:" + str(self.seq) + " - index_old_1=" + str(index_old_1) + " index_old_2=" + str(index_old_2) + " index=" + str(index))
                        
                        # merge
                        depth_jumps[index] = depth_jumps[index_old_1]
                        if depth_jumps[index] < self.max_depth_jump_recognition_count:
                            depth_jumps[index] += self.recognition_increase_rate
                        if index_old_1 != index:
                            depth_jumps[index_old_1] = 0
                        if index_old_2 != index:
                            depth_jumps[index_old_2] = 0
                    elif split_detect and not merge_detect:
                        # split                       
                        index_old = None
                        if index_old_1 != None:
                            index_old = index_old_1
                        else:
                            index_old = index_old_2
                        
                        rospy.logdebug("split - seq:" + str(self.seq) + " - index_old=" + str(index_old) + " index_new_1=" + str(index) + " index_new_2=" + str(index_2))

                        # update count and perform move
                        count = depth_jumps[index_old]
                        if count < self.max_depth_jump_recognition_count:
                            count += 1
                        depth_jumps[index] = count
                        depth_jumps[index_2] = count
                        depth_jumps[index_old] = 0
                    elif index_old_1 != None or index_old_2 != None:
                        # move
                        index_old = index_old_1
                        if index_old == None:
                            index_old = index_old_2
                        rospy.logdebug("move - seq:" + str(self.seq) + " - index_old=" + str(index_old) + " index=" + str(index) + " index_old_1=" + str(index_old_1) + " index_old_2=" + str(index_old_2))
                        depth_jumps[index] = depth_jumps[index_old]
                        if depth_jumps[index] < self.max_depth_jump_recognition_count:
                            depth_jumps[index] += self.recognition_increase_rate
                        if index_old != index:
                            depth_jumps[index_old] = 0
                else:
                    # direct neighbours should not occure this can be used as an indication for a false detection
                    no_neighbours = self._check_for_no_direct_neighbour_depth_discontinuities(index, depth_jumps_detected_from_single_scan, depth_jumps_detected_from_two_scans)
                    if no_neighbours:    
                        # add
                        depth_jumps[index] = 1

            else:
                if depth_jumps[i] > 0:
                    rospy.logdebug("no depth jump position change - seq:" + str(self.seq) + " - index=" + str(i))
                    # check depth jump from t - 1 is visible in single scan analysis
                    if depth_jumps_detected_from_single_scan[i] == 1:
                        if depth_jumps[i] < self.max_depth_jump_recognition_count:
                            depth_jumps[i] += self.recognition_increase_rate
                    else:
                        # Depth jumps get removed over time. If at t -1 a depth jump was detected and at time t no depth jump then decrease.
                        depth_jumps[i] -= self.recognition_decrease_rate
                else:
                    depth_jumps[i] = 0
            
        return depth_jumps     

    def _check_for_split(self, depth_jumps, index, depth_jumps_detected_from_single_scan, index_old_1, index_old_2):
        """
        Determines by if new detected depth discontinuity at index origins from a split of index_old_1 or index_old_2. 

        Parameters: 
        depth_jumps (int[]): depth jumps at t - 1 
        index (int): angele where the depth discontinuity is detected at time t 
        depth_jumps_detected_from_single_scan (int[]): depth jumps calculated from scan_{t} 
        index_old_1 (int): first angle of depth discontinuity at time t-1 that might have split 
        index_old_2 (int): second angle of depth discontinuity at time t-1 that might have split 

        Returns:
        split_detect (bool): tells if a split is detected
        index_2 (int): second angle of split
        """
        # check if there is an other gap detected with in 3 degrees, distance is either 2 or 3 degrees to next gap
        split_detect = False
        split_detect = split_detect or (depth_jumps_detected_from_single_scan[(index + 2) % len(depth_jumps_detected_from_single_scan)] == 1)
        split_detect = split_detect or (depth_jumps_detected_from_single_scan[(index - 2) % len(depth_jumps_detected_from_single_scan)] == 1)
        
        index_2 = None
        # find the second index of the split
        if depth_jumps_detected_from_single_scan[(index + 2) % len(depth_jumps_detected_from_single_scan)] == 1:
            index_2 = (index + 2) % len(depth_jumps_detected_from_single_scan)
        elif depth_jumps_detected_from_single_scan[(index - 2) % len(depth_jumps_detected_from_single_scan)] == 1:
            index_2 = (index - 2) % len(depth_jumps_detected_from_single_scan)

        # check for no depth jump with in 2 degrees left right at t - 1
        if split_detect:
            index_tmp = index_old_1
            if index_tmp == None:
                index_tmp = index_old_2
            
            for j in range(1,4):
                split_detect = split_detect and (depth_jumps[(index_tmp + j) % len(depth_jumps)] == 0)
                split_detect = split_detect and (depth_jumps[(index_tmp - j) % len(depth_jumps)] == 0)

        return split_detect, index_2

    def _check_for_merge(self, index_old_1, index_old_2, index_2):
        """
        Checks if the depth discontinuity at index_old_1 and index_old_2 merged into index_2. 

        Parameter: 
        index_old_1 (int): first angle of depth discontinuity at time t-1 that might have split 
        index_old_2 (int): second angle of depth discontinuity at time t-1 that might have split 
        index_2 (int): angle of depth discontinuity resulting from the merge

        Returns: 
        merge_detect (bool): 
        """
        #two deph_jumps with in a distance of 2 degree to one is a merge
        merge_detect = False
        if index_old_1 != None and index_old_2 != None and index_2 == None:
            diff = abs(index_old_1 - index_old_2)
            if diff > 10:
                diff = 360 - diff
            if diff == 2:
                merge_detect = True

        return merge_detect

    def _check_for_no_direct_neighbour_depth_discontinuities(self, index, depth_jumps_detected_from_single_scan, depth_jumps_detected_from_two_scans):
        """
        Checks if there no consecutive neighbour depth discontinuities with in 3 degrees. 

        Paremeters:
        index (int): index of discontinuity that needs to be checked for neighbours 
        depth_jumps_detected_from_two_scans (int[]): depth jumps calculated from scan_{t} and scan_{t - 1} 
        depth_jumps_detected_from_single_scan (int[]): depth jumps calculated from scan_{t} 

        Returns:
        no_neighbours (bool): true if no neighbours where detected 
        """
        neighbour_count_single_scan = self._get_direct_neighbour_number(depth_jumps_detected_from_single_scan, index, 3)

        neighbour_count_two_scan = self._get_direct_neighbour_number(depth_jumps_detected_from_two_scans, index, 3)
        
        if neighbour_count_single_scan == 0 and neighbour_count_two_scan == 0:
            return True
        else:
            return False

    def _get_direct_neighbour_number(self, discontinuities, index, range):
        """
        Checks if there no consecutive neighbour depth discontinuities with in the given range. 
        
        Parameters: 
        discontinuities (int[]): The array of depth discontinuities 
        index (int): index of discontinuity that needs to be checked for neighbours 
        range (int): number of neigbour elements from index to check 

        Returns: 
        neighbour_count (int): number of the detected neighbours
        """
        neighbour_count = 0
        direct_neighbours = True
        i = 1
        while direct_neighbours and i <= range:
            a = discontinuities[(index - i) % len(discontinuities)] == 1
            b = discontinuities[(index + i) % len(discontinuities)] == 1
            direct_neighbours &= a or b
            if direct_neighbours:
                neighbour_count += 1
            i += 1
        return neighbour_count

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
        for_increment = 1
        search_increment = -1#-1
        depth_jumps_last = self._correction(depth_jumps_last, start, end, for_increment, search_increment, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)

        # 359 -> 180
        start = len(depth_jumps_last) - 1
        end = len(depth_jumps_last) / 2
        for_increment = -1
        search_increment = 1#1
        depth_jumps_last = self._correction(depth_jumps_last, start, end, for_increment, search_increment, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)

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
        for_increment = -1
        search_increment = 1#1
        depth_jumps_last = self._correction(depth_jumps_last, start, end, for_increment, search_increment, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)

        # 180 -> 359
        start = len(depth_jumps_last) / 2
        end = len(depth_jumps_last)
        for_increment = 1
        search_increment = -1#-1
        depth_jumps_last = self._correction(depth_jumps_last, start, end, for_increment, search_increment, depth_jumps_detected_from_two_scans, depth_jumps_detected_from_single_scan)

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
        
        djmsg = DepthJump()
        djmsg.header.stamp = rospy.Time.now()
        djmsg.header.seq = self.seq 
        djmsg.depth_jumps = depth_jumps
        djmsg.range_data = range_data
        djmsg.rotation = rotation
        djmsg.liniear_x = movement
        self.pub_depth_jumps.publish(djmsg)
        self.seq += 1

    def _save_scan_history(self):
        """
        Save scan history to file.
        """
        if self.debug_to_file:
            with open('scan.csv','ab') as f1:
                for tmp in self.scan_history:
                    np.savetxt(f1, tmp.reshape(1, tmp.shape[0]), delimiter=",")

    def _save_dj_1_history(self):
        """
        Save depth jump from one scan history.
        """
        if self.debug_to_file:
            with open('dj_1.csv','ab') as f2:
                for tmp in self.dj_1_history:
                    np.savetxt(f2, tmp.reshape(1, tmp.shape[0]), delimiter=",")

    def _save_dj_2_history(self):
        """
        Save depth jump from two scan history.
        """
        if self.debug_to_file:
            with open('dj_2.csv','ab') as f2:
                for tmp in self.dj_2_history:
                    np.savetxt(f2, tmp.reshape(1, tmp.shape[0]), delimiter=",")

    def _save_depth_jumps_history(self):
        """
        Save depth jumps history.
        """
        if self.debug_to_file:
            with open('depth_jumps.csv','ab') as f2:
                for tmp in self.depth_jumps_history:
                    np.savetxt(f2, tmp.reshape(1, tmp.shape[0]), delimiter=",")

if __name__ == "__main__":
    try:
        gs = DepthJumpSensor()
        gs.run()
    except Exception as ex:
        print(ex.message)
