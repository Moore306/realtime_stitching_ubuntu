#ifndef VIDEO_STITCING_H
#define VIDEO_STITCING_H
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
#endif
using namespace std; 
using namespace cv;


void seamline_fullimage(unsigned char* img, int width,int height,Mat& seam_mask,int &start_x, int & end_x);
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


void opt_track(Mat pre_1,Mat cur_1,Mat pre_2,Mat cur_2,list<Point2f> &kps_1,list<Point2f> & kps_2);

void opt_track(Mat pre_1,Mat cur_1,Mat pre_2,Mat cur_2,vector<Point2f>& prev_key1,vector<Point2f>&prev_key2);


Rect get_overflap_region(vector<Point>& overflap_corners);
void seamline(unsigned char* img, int width,int height,Mat& seam_mask);
void Multiblending(Mat3b& l8u,Mat3b& r8u,Mat& blend_mask,Mat3b& result);