//#include"regions.h"
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


bool PolygonClip(const vector<Point> &poly1,const vector<Point> &poly2, std::vector<Point> &interPoly);
void ClockwiseSortPoints(std::vector<Point> &vPoints);
bool PointCmp(const Point &a,const Point &b,const Point &center);
bool IsPointInPolygon(std::vector<Point> poly,Point pt);
bool IsRectCross(const Point &p1,const Point &p2,const Point &q1,const Point &q2);
bool IsLineSegmentCross(const Point &pFirst1,const Point &pFirst2,const Point &pSecond1,const Point &pSecond2);
bool GetCrossPoint(const Point &p1,const Point &p2,const Point &q1,const Point &q2,long &x,long &y);
Mat Get_blendSize(int& width, int& height, Mat1f& H,Point& offset,vector<Point>& overflap_corners);

bool cvMatEQ(const cv::Mat& data1, const cv::Mat& data2);
void regis(Mat& img1,Mat& img2,vector<Point2f>& obj,vector<Point2f>& scene,Mat &H);

Mat Get_blendSize(int& width, int& height, Mat1f& H,Point2f& offset);
list<Point2f>  keypts1,keypts2;
Mat pre_frame1,pre_frame2;


void opt_track(Mat pre_1,Mat cur_1,Mat pre_2,Mat cur_2,list<Point2f> &kps_1,list<Point2f> & kps_2);

void opt_track(Mat pre_1,Mat cur_1,Mat pre_2,Mat cur_2,vector<Point2f>& prev_key1,vector<Point2f>&prev_key2);



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
    capture1.open("test1.mkv");
    capture2.open("test2.mkv");
//     capture1.set(CV_CAP_PROP_FRAME_WIDTH, 1920);  
//     capture1.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
//     //   
//     capture2.set(CV_CAP_PROP_FRAME_WIDTH, 1920);  
//     capture2.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    namedWindow("img11",0);
    namedWindow("img22",0);
    namedWindow("result",0);
    vector<Point2f> kps1,kps2;
    char info[500];
    int w_index=0;
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

	
	resizeWindow("img11", 640, 480);
	resizeWindow("img22", 640, 480);
	resizeWindow("result", 640, 480);
	

	
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

	Mat1f H;
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	if(index++==0)
	{//orb for first frame 

	
	    regis(img1,img2,kps1,kps2,H);
	    cout<<"匹配点个数 "<<kps1.size()<<endl;
	    if(kps1.size()==0)
		break;
	    
	    Mat img110,img220;
	    img110=img11.clone();
	    img220=img22.clone();
	    for ( auto kp:kps1 )
		cv::circle(img110, kp, 10, cv::Scalar(0, 240, 0), 1);
	    for ( auto kp:kps2 )
		cv::circle(img220, kp, 10, cv::Scalar(240, 240, 0), 1);
	    //imwrite("img110.jpg",img110);
	    //imwrite("img220.jpg",img220);

	    
	}
	else
	{

	    
	    //opt_track(pre_frame1,img11,pre_frame2,img22,keypts1,keypts2);
	    opt_track(pre_frame1,img11,pre_frame2,img22,kps1,kps2);
	    if(kps1.size()<20||index>100)
	    {
		index=0;
		continue;
	    }
	   
	    Mat H1 = findHomography(kps1, kps2, CV_RANSAC);
	    if(H1.empty())
	    {
		index=0;
		continue;
	    }
	    H=H1.clone();
	    Mat img111,img222;
	    img111=img11.clone();
	    img222=img22.clone();
	    for ( auto kp:kps1 )
		cv::circle(img111, kp, 10, Scalar(0, 240, 0), 1);
	    for ( auto kp:kps2 )
		cv::circle(img222, kp, 10, Scalar(240, 240, 0), 1);
	    sprintf(info,"stream 1:tracked points num: %d frame :%d ",kps1.size(),index);
	    putText(img111, info,
		    Point(20, 50),
		    FONT_HERSHEY_COMPLEX, 0.5, // font face and scale
	     Scalar(255, 255, 255), // white
		    1, LINE_AA); // line thickness and type
	    
	    sprintf(info,"stream 2:tracked points num: %d frame :%d ",kps2.size(),index);
	    putText(img222, info,
		    Point(20, 50),
		    FONT_HERSHEY_COMPLEX, 0.5, // font face and scale
	    Scalar(255, 255, 255), // white
		    1, LINE_AA); // line thickness and type
	    imshow("img11",img111);
	    imshow("img22",img222);
	    //imwrite("img11.jpg",img111);
	    //imwrite("img22.jpg",img222);
	
	}
	pre_frame1=img11.clone();
	pre_frame2=img22.clone();
	cout<<"tracked  key points num "<<kps1.size()<<endl;
	
	
	if(kps1.size()<50)
	{
	    //cout<<"kps num is too little  "<<endl;
	    index=0;
	    continue;
	}	
	//cout<<"use "<<H.empty()<<endl;

	//H=H1;;
	Point offset(0,0);
	//cout <<width<<"  "<<height<<endl;
	//Mat1f H1=Get_blendSize(width, height, H,offset);
	vector<Point> overflap_corners;
	////Mat Get_blendSize(int& width, int& height, Mat1f& H,Point& offset,vector<Point>& overfalp_corners);
	
	chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
	
	Mat H1=Get_blendSize(width,height,H,offset,overflap_corners);
	chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t4-t3 );
	cout<<"flapover region  use time："<<time_used.count()<<" seconds."<<endl;
	cout<<"get H1 "<<H1<<endl;
	H=(Mat1f)H1.clone();
	cout<<"success "<<endl;
	//cout<<"another H "<<H<<endl;
	//cout <<width<<"  "<<height<<endl;
	
	if(width<img11.cols||width>5*img11.cols||height<img11.rows||height>5*img11.rows)
	{//index=0;
	    cout <<"continue"<<endl;
	    index=0;
	    continue;
	    
	}
	
	//cout <<"shift "<<offset<<endl;
	//拼接图像
	Mat tiledImg;
	//Mat shftMat=(Mat_<double>(3,3)<<1.0,0,offset.x, 0,1.0,offset.y, 0,0,1.0);
	//warpPerspective(img11,tiledImg,shftMat*H,Size(width+50,height+50));
	warpPerspective(img11,tiledImg,H,Size(width,height));
	Mat tiledImg2=Mat::zeros(tiledImg.rows,tiledImg.cols,tiledImg.type());
	//cout<<tiledImg2.size()<<tiledImg.size()<<endl;
	img22/=2;
	img22.copyTo(Mat(tiledImg2,Rect(abs(offset.x),abs(offset.y),img2.cols,img2.rows)));
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
	cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
	
	
	//imwrite("tied.jpg",tiledImg);
	tiledImg2=tiledImg2/2+tiledImg/2;
	for(auto cor:overflap_corners)
	    circle(tiledImg2, cor, 10, Scalar(240, 240, 0), 1);
	
	imshow("result",tiledImg2);
	imwrite("result.jpg",tiledImg2);
	if(w_index++==50)
	    break;
	

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

      Ptr<ORB> orb = ORB::create(4000, 1.2, 3, 31, 0, 2, 0, 31, 20);  

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

      //cout << "(kpts1: " << kpts1.size() << ") && (kpts2:" << kpts2.size() << ") = goodMatchesKpts: " << goodMatchKpts.size() << endl;
      
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

      //H = findHomography(obj, scene, CV_RANSAC);
      vector<unsigned char> inliersMask(obj.size()); 
      //匹配点对进行RANSAC过滤
      H = findHomography(obj,scene,CV_RANSAC,5,inliersMask);
      vector<Point2f> ransac_obj,ransac_scene;
      for(int i=0;i<inliersMask.size();i++)
      {
	  if(inliersMask[i])
	  {
	      ransac_obj.push_back(obj[i]);
	      ransac_scene.push_back(scene[i]);
	  }
      }
      obj.clear();
      scene.clear();
      obj.assign(ransac_obj.begin(),ransac_obj.end());
      scene.assign(ransac_scene.begin(),ransac_scene.end());
 

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

      //cout<<obj_corners[2]<<endl;

      std::vector<Point2f> scene_corners(4);
      
      perspectiveTransform(obj_corners, scene_corners, H);
      
      
      //储存偏移量
      float w_min=0,h_max=height,w_max=width,h_min=0;

      //cout<<"WH pre   "<<w_max<<" "<<h_max<<"  "<<w_min<<"  "<<h_min<<endl;
      for(int i=0;i<4;i++)
      {
	  //cout<<scene_corners[i]<<"   "<<i<<endl;
	  //cout<<scene_corners[i].x<<endl;

	  if(scene_corners[i].x<w_min)
	      w_min=scene_corners[i].x;
	  if(scene_corners[i].x>w_max)
	      w_max=scene_corners[i].x;
	  if(scene_corners[i].y<h_min)
	      h_min=scene_corners[i].y;
	  if(scene_corners[i].y>h_max)
	      h_max=scene_corners[i].y;
	  
      }

      //cout<<"WH   "<<w_max<<" "<<h_max<<"  "<<w_min<<"  "<<h_min<<endl;

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

      //cout<<"out size "<<width<<"  "<<height<<endl;

      
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

	      //cout<<"################################## "<<status1[i]<<endl;

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
  
  void opt_track(Mat pre_1,Mat cur_1,Mat pre_2,Mat cur_2,vector<Point2f>& prev_key1,vector<Point2f>&prev_key2)
  {
    
     
      vector<unsigned char> status1,status2;
      vector<Point2f>  next_key1,next_key2;
      list<Point2f> kps_1;
      list<Point2f>  kps_2;
      vector<float> error1,error2;
      for(auto kp:prev_key1)
	  kps_1.push_back(kp);
      
      for(auto kp:prev_key2)
	  kps_2.push_back(kp);
      
      //cout<<"before kps num "<<prev_key1.size()<<"  "<<prev_key2.size()<<endl;
      
      calcOpticalFlowPyrLK( pre_1, cur_1, prev_key1, next_key1, status1, error1 );
      calcOpticalFlowPyrLK( pre_2, cur_2, prev_key2, next_key2, status2, error2 );
      int i=0;
      for(auto iter=kps_1.begin(),iter2=kps_2.begin(); iter!=kps_1.end();i++)
      {
	  if ( status1[i]== 0||status2[i]==0)
	  {
	      //cout<<"################################## "<<status2[i]<<endl;
	      iter = kps_1.erase(iter);
	      iter2 = kps_2.erase(iter2);
	  }
	  //cout<<*iter<<"  ------------  "<<next_keypoints[i]<<endl;
	  *iter=next_key1[i];
	  *iter2=next_key2[i];
	
	  iter++;
	  iter2++;
	  
	 
	  
	  
      } 
      prev_key1.clear();
      prev_key2.clear();
      for(auto k:kps_1)
	  prev_key1.push_back(k);
      for(auto k:kps_2)
	  prev_key2.push_back(k);   
      //cout<<"after kps num "<<prev_key1.size()<<"  "<<prev_key2.size()<<endl;

  }
  
 /////////////////////////////////////////////////////////////////////////////
 Mat Get_blendSize(int& width, int& height, Mat1f& H,Point& offset,vector<Point>& overflap_corners)
 {
     //存储左图四角，及其变换到右图位置
     std::vector<Point2f> obj_corners(4);
     obj_corners[0] = Point2f(0, 0); obj_corners[1] = Point2f(width,0);
     obj_corners[2] = Point2f(width, height); obj_corners[3] = Point2f(0, height);
     
     //cout<<obj_corners[2]<<endl;
     
     std::vector<Point2f> scene_corners(4);
     
     perspectiveTransform(obj_corners, scene_corners, H);
     
     
     //储存偏移量
     float w_min=0,h_max=height,w_max=width,h_min=0;
     
     //cout<<"WH pre   "<<w_max<<" "<<h_max<<"  "<<w_min<<"  "<<h_min<<endl;
     for(int i=0;i<4;i++)
     {
	 //cout<<scene_corners[i]<<"   "<<i<<endl;
	 //cout<<scene_corners[i].x<<endl;
	 
	 if(scene_corners[i].x<w_min)
	     w_min=scene_corners[i].x;
	 if(scene_corners[i].x>w_max)
	     w_max=scene_corners[i].x;
	 if(scene_corners[i].y<h_min)
	     h_min=scene_corners[i].y;
	 if(scene_corners[i].y>h_max)
	     h_max=scene_corners[i].y;
	 
     }
     
     //cout<<"WH   "<<w_max<<" "<<h_max<<"  "<<w_min<<"  "<<h_min<<endl;
     
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
     
     //cout<<"out size "<<width<<"  "<<height<<endl;
     
     
     Mat H1 = getPerspectiveTransform(obj_corners, scene_corners);
     
     vector<Point> objs(4),scenes(4);
     for(int i=0;i<4;i++)
     {
	 objs[i].x=obj_corners[i].x-offset.x;
	 objs[i].y=obj_corners[i].y-offset.y;
	 
	 
	 scenes[i].x=scene_corners[i].x;
	 scenes[i].y=scene_corners[i].y;
	 cout<<objs[i]<<"   "<<scenes[i]<<endl;
     }
     
     bool isflap=PolygonClip(objs,scenes,overflap_corners);

     
     for(auto cor:overflap_corners)
	 cout<<"cor "<<cor<<endl;
     return H1;
     
 }
 
 bool PolygonClip(const vector<Point> &poly1,const vector<Point> &poly2, std::vector<Point> &interPoly)
 {
     if (poly1.size() < 3 || poly2.size() < 3)
     {
	 return false;
     }
     
     long x,y;
     //计算多边形交点
     for (int i = 0;i < poly1.size();i++)
     {
	 int poly1_next_idx = (i + 1) % poly1.size();
	 for (int j = 0;j < poly2.size();j++)
	 {
	     int poly2_next_idx = (j + 1) % poly2.size();
	     if (GetCrossPoint(poly1[i],poly1[poly1_next_idx],
		 poly2[j],poly2[poly2_next_idx],
		 x,y))
	     {
		 interPoly.push_back(cv::Point(x,y));
		 cout<<"____ "<<Point(x,y)<<" i,j "<<i<<j<<endl;
	     }
	 }
     }
     cout<<"inter poly.size "<<interPoly.size()<<endl;
     //计算多边形内部点
     for(int i = 0;i < poly1.size();i++)
     {
	 if(IsPointInPolygon(poly2,poly1[i]))
	 {
	     interPoly.push_back(poly1[i]);
	 }
     }
     for (int i = 0;i < poly2.size();i++)
     {
	 if (IsPointInPolygon(poly1,poly2[i]))
	 {
	     interPoly.push_back(poly2[i]);
	 }
     }
     
     if(interPoly.size() <= 0)
	 return false;
     
     //点集排序 
     ClockwiseSortPoints(interPoly);
     return true;
 }
 void ClockwiseSortPoints(std::vector<Point> &vPoints)
 {
     //计算重心
     cv::Point center;
     double x = 0,y = 0;
     for (int i = 0;i < vPoints.size();i++)
     {
	 x += vPoints[i].x;
	 y += vPoints[i].y;
     }
     center.x = (int)x/vPoints.size();
     center.y = (int)y/vPoints.size();
     
     //冒泡排序
     for(int i = 0;i < vPoints.size() - 1;i++)
     {
	 for (int j = 0;j < vPoints.size() - i - 1;j++)
	 {
	     if (PointCmp(vPoints[j],vPoints[j+1],center))
	     {
		 cv::Point tmp = vPoints[j];
		 vPoints[j] = vPoints[j + 1];
		 vPoints[j + 1] = tmp;
	     }
	 }
     }
 }
 bool PointCmp(const Point &a,const Point &b,const Point &center)
 {
     if (a.x >= 0 && b.x < 0)
	 return true;
     if (a.x == 0 && b.x == 0)
	 return a.y > b.y;
     //向量OA和向量OB的叉积
     int det = (a.x - center.x) * (b.y - center.y) - (b.x - center.x) * (a.y - center.y);
     if (det < 0)
	 return true;
     if (det > 0)
	 return false;
     //向量OA和向量OB共线，以距离判断大小
     int d1 = (a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (a.y - center.y);
     int d2 = (b.x - center.x) * (b.x - center.y) + (b.y - center.y) * (b.y - center.y);
     return d1 > d2;
 }
 bool IsRectCross(const Point &p1,const Point &p2,const Point &q1,const Point &q2)
 {
     bool ret = min(p1.x,p2.x) <= max(q1.x,q2.x)    &&
     min(q1.x,q2.x) <= max(p1.x,p2.x) &&
     min(p1.y,p2.y) <= max(q1.y,q2.y) &&
     min(q1.y,q2.y) <= max(p1.y,p2.y);
     return ret;
 }
 //跨立判断
 bool IsLineSegmentCross(const Point &pFirst1,const Point &pFirst2,const Point &pSecond1,const Point &pSecond2)
 {
     long line1,line2;
     line1 = pFirst1.x * (pSecond1.y - pFirst2.y) +
     pFirst2.x * (pFirst1.y - pSecond1.y) +
     pSecond1.x * (pFirst2.y - pFirst1.y);
     line2 = pFirst1.x * (pSecond2.y - pFirst2.y) +
     pFirst2.x * (pFirst1.y - pSecond2.y) + 
     pSecond2.x * (pFirst2.y - pFirst1.y);
     if (((line1 ^ line2) >= 0) && !(line1 == 0 && line2 == 0))
	 return false;
     
     line1 = pSecond1.x * (pFirst1.y - pSecond2.y) +
     pSecond2.x * (pSecond1.y - pFirst1.y) +
     pFirst1.x * (pSecond2.y - pSecond1.y);
     line2 = pSecond1.x * (pFirst2.y - pSecond2.y) + 
     pSecond2.x * (pSecond1.y - pFirst2.y) +
     pFirst2.x * (pSecond2.y - pSecond1.y);
     if (((line1 ^ line2) >= 0) && !(line1 == 0 && line2 == 0))
	 return false;
     return true;
 }
 
 bool GetCrossPoint(const Point &p1,const Point &p2,const Point &q1,const Point &q2,long &x,long &y)
 {
     
     if(IsRectCross(p1,p2,q1,q2))
     {
	 if (IsLineSegmentCross(p1,p2,q1,q2))
	 {
	     //求交点
	     long tmpLeft,tmpRight,b1,b2;
	     b1=(p2.y-p1.y)*p1.x+(p1.x-p2.x)*p1.y;
	     b2=(q2.y-q1.y)*q1.x+(q1.x-q2.x)*q1.y;
	     
	     //tmpLeft = (q2.x - q1.x) * (p1.y - p2.y) - (p2.x - p1.x) * (q1.y - q2.y);
	     //tmpRight = (p1.y - q1.y) * (p2.x - p1.x) * (q2.x - q1.x) + q1.x * (q2.y - q1.y) * (p2.x - p1.x) - p1.x * (p2.y - p1.y) * (q2.x - q1.x);
	     
	     tmpLeft = b2*(p2.x-p1.x)-b1*(q2.x-q1.x);
	     
	     tmpRight = (p2.x-p1.x)*(q2.y-q1.y)-(q2.x-q1.x)*(p2.y-p1.y);
	     
	     
	     x = (int)((double)tmpLeft/(double)tmpRight);
	     
// 	     tmpLeft = (p1.x - p2.x) * (q2.y - q1.y) - (p2.y - p1.y) * (q1.x - q2.x);
// 	     tmpRight = p2.y * (p1.x - p2.x) * (q2.y - q1.y) + (q2.x- p2.x) * (q2.y - q1.y) * (p1.y - p2.y) - q2.y * (q1.x - q2.x) * (p2.y - p1.y); 
	     tmpLeft = b2*(p2.y-p1.y)-b1*(q2.y-q1.y);
	     
	     y = (int)((double)tmpLeft/(double)tmpRight);
	     cout<<"line cross point "<<p1<<p2<<q1<<q2<<x<<y<<endl;
	     return true;
	 }
     }
     
     return false;
 }
 
 bool IsPointInPolygon(std::vector<Point> poly,Point pt)
 {
     int i,j;
     bool c = false;
     for (i = 0,j = poly.size() - 1;i < poly.size();j = i++)
     {
	 if ((((poly[i].y <= pt.y) && (pt.y < poly[j].y)) ||
	     ((poly[j].y <= pt.y) && (pt.y < poly[i].y)))
	     && (pt.x < (poly[j].x - poly[i].x) * (pt.y - poly[i].y)/(poly[j].y - poly[i].y) + poly[i].x))
	 {
	     c = !c;
	 }
     }
     return c;
 }
 