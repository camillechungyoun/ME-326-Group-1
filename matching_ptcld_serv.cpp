#include <ros/ros.h>
#include <ros/package.h>

/*
//This block of Eigen functions aren't required in this script, 
but I personally include this on most applications so I have easy access 
to matrix functionality when needed (similar to python numpy). 
*/
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <eigen_conversions/eigen_msg.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <map> //dictionary equivalent
#include<std_msgs/Header.h>

#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <geometry_msgs/Vector3Stamped.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>


#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>

//include the service for this package
#include <me326_locobot_example/PixtoPoint.h>
#include <ros/console.h>
#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/Point.h>
#include <me326_locobot_example/custommsg.h>

//Setup the class:
class Matching_Pix_to_Ptcld
{
public:
	Matching_Pix_to_Ptcld();

	// Make callback functions for subscribers
	void info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg);
	void depth_callback(const sensor_msgs::Image::ConstPtr& msg);
	void color_image_callback(const sensor_msgs::Image::ConstPtr& msg);
	bool service_callback(me326_locobot_example::PixtoPoint::Request &req, me326_locobot_example::PixtoPoint::Response &res);
	void camera_cube_locator_marker_gen();

private:
	ros::NodeHandle nh;

  // Publisher declarations
	ros::Publisher image_color_filt_pub_;
	ros::Publisher camera_cube_locator_marker_;
	ros::Publisher cube_locations_pub;
	// Subscriber declarations
	ros::Subscriber cam_info_sub_;
	ros::Subscriber depth_sub_;
	ros::Subscriber rgb_image_sub_;
	// Rosservice
	ros::ServiceServer service_;
	//Variables
	geometry_msgs::PointStamped point_3d_cloud_; //Point in pointcloud corresponding to desired pixel
	geometry_msgs::PointStamped point_3d_world; // 3d cube location in world frame
	std::vector<std::vector<geometry_msgs::Point>> pixel_coordinates; // stores all pixel locations. 
	me326_locobot_example::custommsg cube_locations; // stores all the cube locations, in order: red, green, blue, yellow
	std::string color_image_topic_; // this string is over-written by the service request
	std::string depth_image_topic_; // this string is over-written by the service request
	std::string depth_img_camera_info_; // this string is over-written by the service request
	std::string registered_pt_cld_topic_; // this string is over-written by the service request
	image_geometry::PinholeCameraModel camera_model_; //Camera model, will help us with projecting the ray through the depth image
	bool depth_cam_info_ready_; //This will help us ensure we don't ask for a variable before its ready
	// TF Listener
	tf2_ros::Buffer tf_buffer_;
	std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
	tf2_ros::Buffer tf_buffer_world;
	std::unique_ptr<tf2_ros::TransformListener> tf_listener_world;
};

Matching_Pix_to_Ptcld::Matching_Pix_to_Ptcld() 
{
	//Class constructor
	nh = ros::NodeHandle(); //This argument makes all topics internal to this node namespace. //takes the node name (global node handle), If you use ~ then its private (under the node handle name) /armcontroller/ *param*
	//this is how to setup the TF buffer in a class:
	tf_listener_.reset(new tf2_ros::TransformListener(tf_buffer_));
	tf_listener_world.reset(new tf2_ros::TransformListener(tf_buffer_world));

	//ROSparam set variables
	nh.param<std::string>("pt_srv_color_img_topic", color_image_topic_, "/locobot/camera/color/image_raw");
	nh.param<std::string>("pt_srv_depth_img_topic", depth_image_topic_, "/locobot/camera/aligned_depth_to_color/image_raw");
	nh.param<std::string>("pt_srv_depth_img_cam_info_topic", depth_img_camera_info_, "/locobot/camera/aligned_depth_to_color/camera_info");
	nh.param<std::string>("pt_srv_reg_pt_cld_topic", registered_pt_cld_topic_, "/locobot/camera/depth_registered/points");

  // Publisher declarations
	image_color_filt_pub_ = nh.advertise<sensor_msgs::Image>("/locobot/camera/block_color_filt_img",1);
	camera_cube_locator_marker_ = nh.advertise<visualization_msgs::Marker>("/locobot/camera_cube_locator",1);
	cube_locations_pub = nh.advertise<me326_locobot_example::custommsg>("cube_3d_locations",1);
	// Subscriber declarations
	cam_info_sub_ = nh.subscribe(depth_img_camera_info_,1,&Matching_Pix_to_Ptcld::info_callback,this);
	depth_sub_ = nh.subscribe(depth_image_topic_,1,&Matching_Pix_to_Ptcld::depth_callback,this);
	rgb_image_sub_ = nh.subscribe(color_image_topic_,1,&Matching_Pix_to_Ptcld::color_image_callback,this);
	depth_cam_info_ready_ = false; //set this to false so that depth doesn't ask for camera_model_ until its been set
	//Service
	service_ = nh.advertiseService("pix_to_point_cpp", &Matching_Pix_to_Ptcld::service_callback, this);
}

void Matching_Pix_to_Ptcld::camera_cube_locator_marker_gen(){
	visualization_msgs::Marker marker;
	marker.header.frame_id = point_3d_cloud_.header.frame_id;
	marker.header.stamp = ros::Time::now();
	marker.id = 0;
	marker.type = visualization_msgs::Marker::SPHERE;
	// Set the marker scale
	marker.scale.x = 0.05;  //radius of the sphere
	marker.scale.y = 0.05;
	marker.scale.z = 0.05;
	// Set the marker pose
	marker.pose.position.x = point_3d_cloud_.point.x;
	marker.pose.position.y = point_3d_cloud_.point.y;
	marker.pose.position.z = point_3d_cloud_.point.z;
	// Set the marker color
	marker.color.a = 1.0; //transparency
	marker.color.r = 1.0; //red
	marker.color.g = 0.0;
	marker.color.b = 0.0;
	// Publish the marker
	camera_cube_locator_marker_.publish(marker);
}

void Matching_Pix_to_Ptcld::info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg){
	//create a camera model from the camera info
	camera_model_.fromCameraInfo(msg);
	depth_cam_info_ready_ = true;	
}

void Matching_Pix_to_Ptcld::depth_callback(const sensor_msgs::Image::ConstPtr& msg){
	// for each pixel location in pixel_coordinates
	for (int row = 0; row < pixel_coordinates.size(); row++) {
		for (int col = 0; col < pixel_coordinates[row].size(); col++){
			geometry_msgs::Point current_pixel = pixel_coordinates[row][col];
			//Take the depth message, using teh 32FC1 encoding and define the depth pointer
			cv_bridge::CvImageConstPtr cv_ptr;
			try
			{
				cv_ptr = cv_bridge::toCvShare(msg, "32FC1");
			}
			catch (cv_bridge::Exception& e)
			{
				ROS_ERROR("cv_bridge exception: %s", e.what());
				return;
			}

			//Access the pixel of interest
			cv::Mat depth_image = cv_ptr->image;
			float depth_value = depth_image.at<float>(current_pixel.x,current_pixel.y);  // access the depth value of the desired pixel
			//If the pixel that was chosen has non-zero depth, then find the point projected along the ray at that depth value

			if (depth_value == 0)
			{
				ROS_WARN("Skipping cause pixel had no depth");
				return;
			}else{
				if (depth_cam_info_ready_)
				{
					//Pixel has depth, now we need to find the corresponding point in the pointcloud
					//Use the camera model to get the 3D ray for the current pixel
					cv::Point2d pixel(current_pixel.y, current_pixel.x); 
					cv::Point3d ray = camera_model_.projectPixelTo3dRay(pixel);
					//Calculate the 3D point on the ray using the depth value
					cv::Point3d point_3d = ray*depth_value;		
					geometry_msgs::PointStamped point_3d_geom_msg; 
					point_3d_geom_msg.header = msg->header;
					point_3d_geom_msg.point.x = point_3d.x;
					point_3d_geom_msg.point.y = point_3d.y;
					point_3d_geom_msg.point.z = point_3d.z;
					//Transform the point to the pointcloud frame using tf
					std::string point_cloud_frame = camera_model_.tfFrame();
					// Get the camera pose in the desired reference frame
					geometry_msgs::TransformStamped transform;
					try {
						transform = tf_buffer_.lookupTransform(point_cloud_frame, msg->header.frame_id, ros::Time(0)); // target frame, source frame, time
					} catch (tf2::TransformException &ex) {
						ROS_ERROR("%s", ex.what());
					}
					// Transform a point cloud point
					tf2::doTransform(point_3d_geom_msg, point_3d_cloud_, transform); // syntax: (points_in, points_out, transform)

					// transform the points from point_cloud_frame to world frame
					geometry_msgs::TransformStamped transform_camera_to_world;
					try {
						transform_camera_to_world = tf_buffer_world.lookupTransform("locobot/odom", point_cloud_frame, ros::Time(0)); // target frame, source frame, time
					} catch (tf2::TransformException &ex) {
						ROS_ERROR("%s", ex.what());
					}
					tf2::doTransform(point_3d_cloud_, point_3d_world, transform_camera_to_world); // syntax: (points_in, points_out, transform)
				}
			}

			if (row == 0) {
				//cube_locations.red.push_back(point_3d_cloud_.point);
				//ROS_INFO("red");
				cube_locations.red_x.push_back(point_3d_world.point.x);
				cube_locations.red_y.push_back(point_3d_world.point.y);
				cube_locations.red_z.push_back(point_3d_world.point.z);
			} else if (row == 1) {
				cube_locations.green_x.push_back(point_3d_world.point.x);
				cube_locations.green_y.push_back(point_3d_world.point.y);
				cube_locations.green_z.push_back(point_3d_world.point.z);
				//ROS_INFO("green");
			} else if (row == 2) {
				cube_locations.blue_x.push_back(point_3d_world.point.x);
				cube_locations.blue_y.push_back(point_3d_world.point.y);
				cube_locations.blue_z.push_back(point_3d_world.point.z);
				//ROS_INFO("blue");
			} else {
				cube_locations.yellow_x.push_back(point_3d_world.point.x);
				cube_locations.yellow_y.push_back(point_3d_world.point.y);
				cube_locations.yellow_z.push_back(point_3d_world.point.z);
				//ROS_INFO("yellow");
			}
		}
	}
	cube_locations_pub.publish(cube_locations);
}

void Matching_Pix_to_Ptcld::color_image_callback(const sensor_msgs::Image::ConstPtr& msg){
	//convert sensor_msgs image to opencv image : http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages
	cv_bridge::CvImagePtr color_img_ptr;
	try
	{
	  color_img_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8); //accesses image through color_img_ptr->image
	}
	catch (cv_bridge::Exception& e)
	{
	  ROS_ERROR("cv_bridge exception: %s", e.what());
	  return;
	}
	//Convert opencv color imgage to HSV:
	cv::Mat hsv; 
	cv::cvtColor(color_img_ptr->image, hsv, cv::COLOR_RGB2HSV); //example: https://cppsecrets.com/users/203110310511410511510410011599115495764103109971051084699111109/C00-OpenCv-cvcvtColor.php, and https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
	// Now generate lower and upper bounds of HSV to filter out cube of particular color (demonstrated - red filter)  
  	cv::Mat lower_bound = cv::Mat::zeros(hsv.size(), hsv.type());
  	cv::Mat upper_bound = cv::Mat::zeros(hsv.size(), hsv.type());

	for (int i = 0; i < 4; i++) { // loop through the 4 colors
		if (i == 0) {
			// red 
  			lower_bound.setTo(cv::Scalar(0, 100, 20));
  			upper_bound.setTo(cv::Scalar(5, 255, 255));
		} else if (i == 1) {
   			// green
   			lower_bound.setTo(cv::Scalar(36, 50, 30));
   			upper_bound.setTo(cv::Scalar(89, 255, 255));
		} else if (i == 2) {
			// blue
   			lower_bound.setTo(cv::Scalar(107, 90, 10));
   			upper_bound.setTo(cv::Scalar(130, 255, 200));
		} else {
   			// yellow
			lower_bound.setTo(cv::Scalar(20, 70, 20));
			upper_bound.setTo(cv::Scalar(35, 255, 255));
		}
		// Now generate and filter to make the mask:
		cv::Mat mask;
		cv::inRange(hsv, lower_bound, upper_bound, mask);

		// ***NOW THIS CODE BELOW MAKES THE VERY STRONG ASSUMPTION THAT THERE IS ONLY ONE CUBE IN THE FIELD OF VIEW OF THE DESIRED COLOR IN THE MASK - IT AVERAGES THE MASK PIXELS TO FIND THE CENTER	
		cv::Mat mask_img; //this is the result 	//Apply the mask; black region in the mask is 0, so when multiplied with original image removes all non-selected color
		cv::bitwise_and(color_img_ptr->image, color_img_ptr->image, mask_img, mask); //https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga60b4d04b251ba5eb1392c34425497e14  
		
		cv::Mat greyMat;
		cv::cvtColor(mask_img, greyMat, cv::COLOR_BGR2GRAY); // convert from color to black and grey 
		cv::Mat bw = greyMat > .01; // convert from grey to black and white 
		//std::cout << "bw sum = " << std::endl << " " << s << std::endl;

		cv::SimpleBlobDetector::Params params;
		params.filterByColor = true;
		params.blobColor = 255; // 255 = white, 0 = black. block should be white
		cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params); 
		std::vector<cv::KeyPoint> keypoints; 
		detector->detect(bw, keypoints); // populates keypoints
		//ROS_INFO("numkeypoints = %lu", keypoints.size());

		float my_x;
		float my_y;
		// loop through each of the keypoints (blocks) of the selected color
		std::vector<geometry_msgs::Point> points_one_color;
		for (int i = 0; i < keypoints.size(); i++) { // need to reverse x and y because type cv::Point2f order is (column,row)
			my_x = keypoints[i].pt.y; 
			my_y = keypoints[i].pt.x;
			//ROS_INFO("D");

			// Turn the average pixel location white; Make the center point pixel bright so it shows up in this image
			int x = (int) my_x;
			int y = (int) my_y;
			mask_img.at<cv::Vec3b>(x, y) = cv::Vec3b(255, 255, 255);

			geometry_msgs::Point pix_point;
			pix_point.x = my_x;
			pix_point.y = my_y;	
			points_one_color.push_back(pix_point);
		}
		pixel_coordinates.push_back(points_one_color);
	}
}


bool Matching_Pix_to_Ptcld::service_callback(me326_locobot_example::PixtoPoint::Request &req, me326_locobot_example::PixtoPoint::Response &res){
	// the topic for the rgb_img should be set as a rosparam when the file is launched (this can be done in the launch file, it is not done here since the subscriber is started with the class object instantiation)
	res.ptCld_point = point_3d_cloud_; //send the point back as a response	
	return true;
}

int main(int argc, char **argv)
{
  ros::init(argc,argv,"matching_ptcld_serv");
  ros::NodeHandle nh("~");
  Matching_Pix_to_Ptcld ctd_obj;
  ros::spin();
  return 0;
}