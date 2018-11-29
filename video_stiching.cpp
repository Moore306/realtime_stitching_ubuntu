#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>

#include<stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include<opencv2/imgproc.hpp>

#include "opencv2/opencv.hpp" 


using namespace std; 
using namespace cv;
cv::Mat color, depth, last_color;

bool cvMatEQ(const cv::Mat& data1, const cv::Mat& data2);
void regis(Mat& img1,Mat& img2,vector<Point2f>& obj,vector<Point2f>& scene,Mat &H);

Mat Get_blendSize(int& width, int& height, Mat1f& H,Point2f& offset);
list<Point2f>  keypts1,keypts2;
Mat pre_frame1,pre_frame2;


void opt_track(Mat pre_1,Mat cur_1,Mat pre_2,Mat cur_2,list<Point2f> &kps_1,list<Point2f> & kps_2);


int main()
{
    
    // 配置图像路径
    //Mat img11 = imread("../c.png");
    //Mat img22 = imread("../d.png");
    //cout<<img11.size()<<endl;
    int width,height;
    Mat img11,img22;
    Mat img1,img2;
    VideoCapture capture1,capture2;
    int index=0;
    capture1.open(2);
    capture2.open(1);
    //    capture1.set(CV_CAP_PROP_FRAME_WIDTH, 1920);  
    //   capture1.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    //   
    //    capture2.set(CV_CAP_PROP_FRAME_WIDTH, 1920);  
    //   capture2.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    
    while(true)
    {
	
	//if(index++!=0)
	//break;
	if(waitKey(1) >= 0)
	{ //break;
	    
	    imwrite("a.jpg",img11);
	    imwrite("b.jpg",img22);
	    
	}
	
	capture1>>img11;
	capture2>>img22;
	namedWindow("img11",0);
	namedWindow("img22",0);
	resizeWindow("img11", 640, 480);
	resizeWindow("img22", 640, 480);
	imshow("img11",img11);
	imshow("img22",img22);
	
	width=img11.cols;
	height=img11.rows;
	//continue;
	cvtColor(img11,img1,CV_BGR2GRAY);
	cvtColor(img22,img2,CV_BGR2GRAY);
	// 判断输入图像是否读取成功
	if (img1.empty() || img2.empty() || img1.channels() != 1 || img2.channels() != 1)
	{
	    cout << "Input Image is nullptr or the image channels is not gray!" << endl;
	    //system("pause");
	}
	vector<Point2f> kps1,kps2;
	Mat1f H;
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	if(index++==0)
	{//orb for first frame 
	    keypts1.clear();
	    keypts2.clear();
	    regis(img1,img2,kps1,kps2,H);
	    
	    for(int i=0;i<kps1.size();i++)
	    {
		keypts1.push_back(kps1[i]);
		keypts2.push_back(kps2[i]);
	    }
	    
	    
	}
	else
	{
	    kps1.clear();kps2.clear();
	    opt_track(pre_frame1,img11,pre_frame2,img22,keypts1,keypts2);
	    for(auto iter=keypts1.begin(),iter2=keypts2.begin();iter!=keypts1.end();iter++,iter2++)
	    {
		kps1.push_back(*iter);
		kps2.push_back(*iter2);
	    }
	    Mat H1 = findHomography(kps1, kps2, CV_RANSAC);
	    H=H1.clone();
	
	}
	
	
	if(kps1.size()<10)
	{
	    cout<<"kps1"<<endl;
	    index=0;
	    continue;
	}
	cout<<H<<endl;
	//H=H1;;
	Point2f offset(0,0);
	cout <<width<<"  "<<height<<endl;
	Mat1f H1=Get_blendSize(width, height, H,offset);
	H=H1.clone();
	cout<<"another H "<<H<<endl;
	cout <<width<<"  "<<height<<endl;
	
	if(width<img11.cols||width>2*img11.cols||height<img11.rows||height>2*img11.rows)
	{//index=0;
	    cout <<"continue"<<endl;
	    continue;
	    
	}
	
	
	cout <<"shift "<<offset<<endl;
	//拼接图像
	Mat tiledImg;
	//Mat shftMat=(Mat_<double>(3,3)<<1.0,0,offset.x, 0,1.0,offset.y, 0,0,1.0);
	//warpPerspective(img11,tiledImg,shftMat*H,Size(width+50,height+50));
	warpPerspective(img11,tiledImg,H,Size(width,height));
	Mat tiledImg2=Mat::zeros(tiledImg.rows,tiledImg.cols,tiledImg.type());
	cout<<tiledImg2.size()<<tiledImg.size()<<endl;
	img22/=2;
	img22.copyTo(Mat(tiledImg2,Rect(abs(offset.x),abs(offset.y),img2.cols,img2.rows)));
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
	cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
	//imwrite("tied.jpg",tiledImg);
	tiledImg2=tiledImg2/2+tiledImg/2;
	resizeWindow("result", 640, 480);
	imshow("result",tiledImg2);
	imwrite("result.jpg",tiledImg2);
	
	pre_frame1=img11.clone();
	pre_frame2=img22.clone();
    }
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
  
  void regis(Mat& img1,Mat& img2,vector<Point2f>& obj,vector<Point2f>& scene,Mat &H)
  {
      obj.clear();
      scene.clear();
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
      cout << "(kpts1: " << kpts1.size() << ") && (kpts2:" \
      << kpts2.size() << ") = goodMatchesKpts: " << goodMatchKpts.size() << endl;
      
      //waitKey(0);
      
      // RANSAC Geometric Verification
      if (goodMatchKpts.size() < 4)
      {
	  cout << "The Match Kpts' Size is less than Four to estimate!" << endl;
	  return;
      }
      
      
      for (unsigned int i = 0; i < goodMatchKpts.size(); ++i)
      {
	  obj.push_back(kpts1[goodMatchKpts[i].queryIdx].pt);
	  scene.push_back(kpts2[goodMatchKpts[i].trainIdx].pt);
      }
      // 估计Two Views变换矩阵
      H = findHomography(obj, scene, CV_RANSAC);
      vector<Point2f> obj_corners(4), scene_corners(4);
      obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img1.cols, 0);
      obj_corners[2] = cvPoint(img1.cols, img1.rows); obj_corners[3] = cvPoint(0, img1.rows);
      // 点集变换标出匹配重复区域
      //perspectiveTransform(obj_corners, scene_corners, H);
  }
  Mat Get_blendSize(int& width, int& height, Mat1f& H,Point2f& offset) {
      //存储左图四角，及其变换到右图位置
      std::vector<Point2f> obj_corners(4);
      obj_corners[0] = Point2f(0, 0); obj_corners[1] = Point2f(width,0);
      obj_corners[2] = Point2f(width, height); obj_corners[3] = Point2f(0, height);
      cout<<obj_corners[2]<<endl;
      std::vector<Point2f> scene_corners(4);
      
      perspectiveTransform(obj_corners, scene_corners, H);
      
      
      //储存偏移量
      float w_min=0,h_max=height,w_max=width,h_min=0;
      cout<<"WH pre   "<<w_max<<" "<<h_max<<"  "<<w_min<<"  "<<h_min<<endl;
      for(int i=0;i<4;i++)
      {
	  cout<<scene_corners[i]<<"   "<<i<<endl;
	  cout<<scene_corners[i].x<<endl;
	  if(scene_corners[i].x<w_min)
	      w_min=scene_corners[i].x;
	  if(scene_corners[i].x>w_max)
	      w_max=scene_corners[i].x;
	  if(scene_corners[i].y<h_min)
	      h_min=scene_corners[i].y;
	  if(scene_corners[i].y>h_max)
	      h_max=scene_corners[i].y;
	  
      }
      cout<<"WH   "<<w_max<<" "<<h_max<<"  "<<w_min<<"  "<<h_min<<endl;
      //新建一个矩阵存储配准后四角的位置
      width = w_max-w_min+50;
      //int height= img1.rows;
      height = h_max-h_min+50;
      
      offset.x=w_min;
      offset.y=h_min;
      for(int i=0;i<4;i++)
      {
	  scene_corners[i].x-=offset.x;
	  scene_corners[i].y-=offset.y;
      }
      cout<<"out size "<<width<<"  "<<height<<endl;
      
      Mat H1 = getPerspectiveTransform(obj_corners, scene_corners);
      return H1;
      
      
  }

  
  void opt_track(Mat pre_1,Mat cur_1,Mat pre_2,Mat cur_2,list<Point2f> &kps_1,list<Point2f> & kps_2)
  {
      vector<unsigned char> status1,status2;
      vector<Point2f>  prev_key1,prev_key2,next_key1,next_key2;
      vector<float> error1,error2;
      for(auto kp:kps_1)
          prev_key1.push_back(kp);
      
      for(auto kp:kps_2)
	  prev_key2.push_back(kp);
      
      
	  
      calcOpticalFlowPyrLK( pre_1, cur_1, prev_key1, next_key1, status1, error1 );
      calcOpticalFlowPyrLK( pre_2, cur_2, prev_key2, next_key2, status2, error2 );
      int i=0;
    
      for ( auto iter=kps_1.begin(),iter2=kps_2.begin(); iter!=kps_1.end(); i++)
      {
	  if ( status1[i]== 0 ||status2[i]==0)
	  {
	      cout<<"################################## "<<status1[i]<<endl;
	      iter = kps_1.erase(iter);
	      iter2 = kps_2.erase(iter2);
	  }
	  //cout<<*iter<<"  ------------  "<<next_keypoints[i]<<endl;
	  *iter = next_key1[i];
	  *iter2=next_key2[i];
	  iter++;
	  iter2++;
      } 
     
     
      
  }