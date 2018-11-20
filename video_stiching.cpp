#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
using namespace std; 
#include<stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
using namespace cv;
cv::Mat color, depth, last_color;
bool cvMatEQ(const cv::Mat& data1, const cv::Mat& data2);
int main( int argc, char** argv )
{
  
    cv::Mat color,last_color,color_show;
    VideoCapture capture;
    capture.open(0);
    int index=0;
   // capture.set(CV_CAP_PROP_FRAME_WIDTH, 1920);  
   // capture.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    if(capture.isOpened())
    {
      
	list< cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list
	vector<cv::Point2f> next_keypoints; 
	vector<cv::Point2f> prev_keypoints;
	vector<unsigned char> status;
	vector<float> error; 
        cout << "Capture is opened" << endl;
	vector<Point2f> * pre_frame,next_frame;
	vector<cv::KeyPoint> kps;
	char info[500];
        while(true)
        {
	    status.clear();
	    
	    error.clear();
            capture.read(color);
	    cout<<"video size "<<color.size()<<endl;
	    
 	    if(waitKey(1) >= 0)
	        break;
	    if (index++ ==0 )
	    {
		// 对第一帧提取FAST特征点
		
		cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
		detector->detect( color, kps );
		for(auto kp:kps)
		{
		  prev_keypoints.push_back(kp.pt);
		  keypoints.push_back(kp.pt);
		}
		
		
		cout<<keypoints.size()<<endl;
		last_color = color.clone();
		continue;
	    }
	    if ( color.data==nullptr )
		continue;
	    // 对其他帧用LK跟踪特征点
	   
	    cout<<"prev_keypoints size  "<<prev_keypoints.size()<<endl;
	  
	    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();  
	    if(cvMatEQ(last_color,color))
	      cout<<"true"<<endl;
	      
	    cv::calcOpticalFlowPyrLK( last_color, color, prev_keypoints, next_keypoints, status, error );
	    imshow("differ",last_color-color);
	    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
	    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
	    // 把跟丢的点删掉
	    int i=0; 
	    for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)
	    {
		if ( status[i]== 0 )
		{
		    cout<<"################################## "<<status[i]<<endl;
		    iter = keypoints.erase(iter);
		    continue;
		}
		cout<<*iter<<"  ------------  "<<next_keypoints[i]<<endl;
		*iter = next_keypoints[i];
		iter++;
	    }
	    cout<<"tracked keypoints: "<<keypoints.size()<<endl;
	    if (keypoints.size() == 0)
	    {
		cout<<"all keypoints are lost."<<endl;
		break; 
	    }
	    // 画出 keypoints
	    prev_keypoints.clear();
	    next_keypoints.clear();
	    for(auto kp:keypoints)
	      prev_keypoints.push_back(kp);
	    color_show=color.clone();
	    for ( auto kp:keypoints )
	        //drawKeypoints(color, kps,color_show, Scalar(0,255,255));
		cv::circle(color_show, kp, 1, cv::Scalar(0, 240, 0), 1);
	    sprintf(info,"tracked points num: %d cost time:%f frame :%d ",keypoints.size(),time_used.count(),index);
	    putText(color_show, info,
            Point(20, 50),
            FONT_HERSHEY_COMPLEX, 0.5, // font face and scale
            Scalar(255, 255, 255), // white
            1, LINE_AA); // line thickness and type
	    cv::imshow("corners", color_show);
	    if(index%10==0)
	    {
	      sprintf(info,"flow/flow_%3d.jpg",index);
	      imwrite(info,color_show);
	    }
	  
	    last_color = color.clone();
	
	} 
    }
   
    return 0;
}
bool cvMatEQ(const cv::Mat& data1, const cv::Mat& data2)
{
    bool success = true;
    // check if is multi dimensional
    if(data1.dims > 2 || data2.dims > 2)
    {
      if( data1.dims != data2.dims || data1.type() != data2.type() )
      {
        return false;
      }
      for(long dim = 0; dim < data1.dims; dim++){
        if(data1.size[dim] != data2.size[dim]){
          return false;
        }
      }
    }
    else
    {
      if(data1.size() != data2.size() || data1.channels() != data2.channels() || data1.type() != data2.type()){
        return false;
      }
    }
    int nrOfElements = data1.total()*data1.elemSize1();
    //bytewise comparison of data
    int cnt = 0;
    for(cnt = 0; cnt < nrOfElements && success; cnt++)
    {
      if(data1.data[cnt] != data2.data[cnt]){
        success = false;
      }
    }
    return success;
  }
