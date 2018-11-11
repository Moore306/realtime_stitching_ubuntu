#include <iostream>
#include "opencv2/opencv.hpp" 
#include <opencv2/features2d/features2d.hpp>
//#include<ldb>

using namespace std;
using namespace cv;

int main(void)
{
    // 配置图像路径
    Mat img1 = imread("../c.png", 0);
    Mat img2 = imread("../d.png", 0);
    // 判断输入图像是否读取成功
    if (img1.empty() || img2.empty() || img1.channels() != 1 || img2.channels() != 1)
    {
        cout << "Input Image is nullptr or the image channels is not gray!" << endl;
        system("pause");
    }
    // ORB算法继承Feature2D基类
    Ptr<ORB> orb = ORB::create(1000, 1.2, 8, 31, 0, 2, 0, 31, 20);  
    // 调整精度，值越小点越少，越精准
    vector<KeyPoint> kpts1, kpts2;
    // 特征点检测算法...
    orb->detect(img1, kpts1);
    orb->detect(img2, kpts2);

    // 特征点描述算法...
    Mat desc1, desc2;

    bool SelectiveDescMethods = true;
    // 默认选择BRIEF描述符
    if (SelectiveDescMethods) 
    {
        // ORB 算法中默认BRIEF描述符
        orb->compute(img1, kpts1, desc1);
        orb->compute(img2, kpts2, desc2);
    }
 
    // 粗精匹配数据存储结构
    vector< vector<DMatch>> matches;
    vector<DMatch> goodMatchKpts;
    // Keypoint Matching...
    DescriptorMatcher *pMatcher = new BFMatcher(NORM_HAMMING, false);
    pMatcher->knnMatch(desc1, desc2, matches, 2);
    // 欧式距离度量  阈值设置为0.8
    for (unsigned int i = 0; i < matches.size(); ++i)
    {
        if (matches[i][0].distance < 0.8*matches[i][1].distance)
        {
            goodMatchKpts.push_back(matches[i][0]);
        }
    }
    // 显示匹配点对
    Mat show_match;
    drawMatches(img1, kpts1, img2, kpts2, goodMatchKpts, show_match);

    // 显示输出
    ostringstream s_time;
    s_time << time;
    imshow("ORB_Algorithms_" + s_time.str(), show_match);

    cout << "(kpts1: " << kpts1.size() << ") && (kpts2:" \
         << kpts2.size() << ") = goodMatchesKpts: " << goodMatchKpts.size() << endl;

    waitKey(0);

    // RANSAC Geometric Verification
    if (goodMatchKpts.size() < 4)
    {
        cout << "The Match Kpts' Size is less than Four to estimate!" << endl;
        return 0;
    }

    vector<Point2f> obj, scene;
    for (unsigned int i = 0; i < goodMatchKpts.size(); ++i)
    {
        obj.push_back(kpts1[goodMatchKpts[i].queryIdx].pt);
        scene.push_back(kpts2[goodMatchKpts[i].trainIdx].pt);
    }
    // 估计Two Views变换矩阵
    Mat H = findHomography(obj, scene, CV_RANSAC);
    vector<Point2f> obj_corners(4), scene_corners(4);
    obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img1.cols, 0);
    obj_corners[2] = cvPoint(img1.cols, img1.rows); obj_corners[3] = cvPoint(0, img1.rows);
    // 点集变换标出匹配重复区域
    perspectiveTransform(obj_corners, scene_corners, H);

    line(show_match, scene_corners[0] + Point2f(img1.cols, 0), scene_corners[1] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
    line(show_match, scene_corners[1] + Point2f(img1.cols, 0), scene_corners[2] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
    line(show_match, scene_corners[2] + Point2f(img1.cols, 0), scene_corners[3] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
    line(show_match, scene_corners[3] + Point2f(img1.cols, 0), scene_corners[0] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);

    imshow("Match End", show_match);
    imwrite("img_boat15.jpg", show_match);
    waitKey(0);
    system("pause");
    return 0;
}


// #include <iostream>
// #include <fstream>
// #include <list>
// #include <vector>
// #include <chrono>
// using namespace std; 
// #include<stdio.h>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/features2d/features2d.hpp>
// #include <opencv2/video/tracking.hpp>
// #include<opencv2/imgproc.hpp>
// using namespace cv;
// #include "opencv2/opencv.hpp" 
// cv::Mat color, depth, last_color;
// //bool cvMatEQ(const cv::Mat& data1, const cv::Mat& data2);
// void regis(Mat & img1,Mat & img2,Mat & img_out);
// int main()
// {
//   vector<Mat> streams;
//   vector<Mat> Hs;
//   vector<vector<Point2f>> prev_keypoints,curr_keypoints;
//   Mat result,img;
//   char  win_info[10][100];
//   int index=0;
//   vector<VideoCapture> capture;
//   VideoCapture capture1,capture2,capture3,capture4;
//   int video_num=2;
//  
//   capture1.open(1);
//   capture2.open(2);
//   capture1.set(CV_CAP_PROP_FRAME_WIDTH, 1920);  
//   capture1.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
//   
//   capture2.set(CV_CAP_PROP_FRAME_WIDTH, 1920);  
//   capture2.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
//   
//   
//   capture.push_back(capture1);
//   capture.push_back(capture2);
//   
//   for(int i=0;i<video_num;i++)
//   {
//     
//     if(capture[i].isOpened())
//     { 
//       cout<<"capture "<<i<<endl;
//       capture[i]>>img;
//     }
//     streams.push_back(img.clone());
//   }
//   
//   while(true)
//   {
//       //cout<<index++<<endl;
//       if(waitKey(1) >= 0)
// 	        break;
//       for(int i=0;i<video_num;i++)
//       {
// 	capture[i].read(img);
// 	cout<<img.size()<<endl;
// 	streams[i]=img.clone();
// 	sprintf(win_info[i],"stream :%d ",i);
// 	cout<<win_info[i]<<endl;
// 	imshow(win_info[i],streams[i]);
//       }
//   }
//   
//   
//   
//   
// }
// void regis(Mat & img1,Mat & img2,Mat & img_out,vector<Point2f> kp1,vector<Point2f> kp2)
// {
//   
// }






// int main( int argc, char** argv )
// {
//   
//     cv::Mat color,last_color,color_show,result;
//     VideoCapture capture;
//     capture.open(0);
//     int index=0;

//     if(capture.isOpened())
//     {
//       
// 	list< cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list
// 	vector <Point2f>  keypoints_next;
// 	vector<cv::Point2f> next_keypoints; 
// 	vector<cv::Point2f> prev_keypoints;
// 	vector<unsigned char> status;
// 	vector<float> error; 
//         cout << "Capture is opened" << endl;
// 	vector<Point2f> * pre_frame,next_frame;
// 	vector<cv::KeyPoint> kps;
// 	char info[500];
//         while(true)
//         {
// 	    status.clear();
// 	    
// 	    error.clear();
//             capture.read(color);
// 	    cout<<"video size "<<color.size()<<endl;
// 	    
//  	    if(waitKey(1) >= 0)
// 	        break;
// 	    if (index++ ==0 )
// 	    {
// 		// 对第一帧提取FAST特征点
// 		//color=imread("../c.png");
// 		cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
// 		detector->detect( color, kps );
// 		for(auto kp:kps)
// 		{
// 		  prev_keypoints.push_back(kp.pt);
// 		  keypoints.push_back(kp.pt);
// 		}
// 		
// 		
// 		cout<<keypoints.size()<<endl;
// 		last_color = color.clone();
// 		continue;
// 	    }
// 	    if ( color.data==nullptr )
// 		continue;
// 	    // 对其他帧用LK跟踪特征点
// 	    //color=imread("../d.png");
// 	    cout<<"prev_keypoints size  "<<prev_keypoints.size()<<endl;
// 	  
// 	    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();  
// 	    if(cvMatEQ(last_color,color))
// 	      cout<<"true"<<endl;
// 	      
// 	    cv::calcOpticalFlowPyrLK( last_color, color, prev_keypoints, next_keypoints, status, error );
// 	    //imshow("differ",last_color-color);
// 	    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
// 	    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
// 	    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
// 	    // 把跟丢的点删掉
// 	    int i=0; 
// 	    keypoints_next.clear();
// 	    for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)
// 	    {
// 		if ( status[i]== 0 )
// 		{
// 		    cout<<"################################## "<<status[i]<<endl;
// 		    iter = keypoints.erase(iter);
// 		    continue;
// 		}
// 		//cout<<*iter<<"  ------------  "<<next_keypoints[i]<<endl;
// 		*iter = next_keypoints[i];
// 		keypoints_next.push_back(next_keypoints[i]);
// 		iter++;
// 	    }
// 	    cout<<"tracked keypoints: "<<keypoints.size()<<endl;
// 	    if (keypoints.size() == 0)
// 	    {
// 		cout<<"all keypoints are lost."<<endl;
// 		break; 
// 	    }
// 	    
// 	    // 画出 keypoints
// 	    prev_keypoints.clear();
// 	    next_keypoints.clear();
// 	    for(auto kp:keypoints)
// 	      prev_keypoints.push_back(kp);
// 	    Mat H = findHomography(prev_keypoints,keypoints_next);  
// 	    cout<<H<<endl;
// 	    warpPerspective(last_color, result, H,color.size());
// 	    result=result/2+color/2;
// 	    cout<<result.size()<<endl;
// 	    imwrite("color.jpg",color);
// 	    imwrite("last_color.jpg",last_color);
// 	    imshow("result",result);
// 	    imwrite("result.jpg",result);
// 	    color_show=color.clone();
// 	    for ( auto kp:keypoints )
// 	        //drawKeypoints(color, kps,color_show, Scalar(0,255,255));
// 		cv::circle(color_show, kp, 1, cv::Scalar(0, 240, 0), 1);
// 	    sprintf(info,"tracked points num: %d cost time:%f frame :%d ",keypoints.size(),time_used.count(),index);
// 	    putText(color_show, info,
//             Point(20, 50),
//             FONT_HERSHEY_COMPLEX, 0.5, // font face and scale
//             Scalar(255, 255, 255), // white
//             1, LINE_AA); // line thickness and type
// 	    cv::imshow("corners", color_show);
// 	  
// 	    last_color = color.clone();
// 	    //if(index==2)
// 	      //break;
// 	
// 	} 
//     }
//    
//     return 0;
// }
// bool cvMatEQ(const cv::Mat& data1, const cv::Mat& data2)
// {
//     bool success = true;
//     // check if is multi dimensional
//     if(data1.dims > 2 || data2.dims > 2)
//     {
//       if( data1.dims != data2.dims || data1.type() != data2.type() )
//       {
//         return false;
//       }
//       for(long dim = 0; dim < data1.dims; dim++){
//         if(data1.size[dim] != data2.size[dim]){
//           return false;
//         }
//       }
//     }
//     else
//     {
//       if(data1.size() != data2.size() || data1.channels() != data2.channels() || data1.type() != data2.type()){
//         return false;
//       }
//     }
//     int nrOfElements = data1.total()*data1.elemSize1();
//     //bytewise comparison of data
//     int cnt = 0;
//     for(cnt = 0; cnt < nrOfElements && success; cnt++)
//     {
//       if(data1.data[cnt] != data2.data[cnt]){
//         success = false;
//       }
//     }
//     return success;
//   }
