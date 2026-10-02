#!/usr/bin/env python3

import rospy
import message_filters
import rosbag
from sensor_msgs.msg import PointCloud2

class DataSaverNode:
    def __init__(self):
        rospy.init_node('synced_data_saver', anonymous=True)
        
        # Parameters
        self.save_interval = 10  # Changed to 10 so you see results faster!
        self.counter = 0
        self.bag_filename = '/home/apyszko/catkin_ws/src/RMS/synchronized_subset.bag'
        
        # Open a new bag file to write the synced data
        self.bag = rosbag.Bag(self.bag_filename, 'w')
        rospy.loginfo(f"Opened new bag file for saving: {self.bag_filename}")
        rospy.on_shutdown(self.close_bag)

        # UPDATED TOPIC NAMES HERE!
        raw_sub = message_filters.Subscriber('/mulran/velo/pointclouds', PointCloud2)
        rms_sub = message_filters.Subscriber('/mulran/velo/pointclouds_rms', PointCloud2)

        # Synchronize the two topics
        self.ts = message_filters.ApproximateTimeSynchronizer([raw_sub, rms_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.sync_callback)
        
        rospy.loginfo("Listening on /mulran/velo/pointclouds and /mulran/velo/pointclouds_rms...")

    def sync_callback(self, raw_msg, rms_msg):
        self.counter += 1
        
        if self.counter % self.save_interval == 0:
            rospy.loginfo(f"Match found! Saving pair #{self.counter} to bag...")
            self.bag.write('/saved/raw_points', raw_msg, raw_msg.header.stamp)
            self.bag.write('/saved/rms_points', rms_msg, rms_msg.header.stamp)

    def close_bag(self):
        rospy.loginfo("Closing bag file safely...")
        self.bag.close()

if __name__ == '__main__':
    try:
        DataSaverNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass