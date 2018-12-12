//#include"regions.h"
#include<video_stitching.h>
class LaplacianBlending {
private:
    Mat_<Vec3f> left;
    Mat_<Vec3f> right;
    Mat_<float> blendMask;
    
    vector<Mat_<Vec3f> > leftLapPyr,rightLapPyr,resultLapPyr;//Laplacian Pyramids
    Mat leftHighestLevel, rightHighestLevel, resultHighestLevel;
    vector<Mat_<Vec3f> > maskGaussianPyramid; //masks are 3-channels for easier multiplication with RGB
    
    int levels;
    
    void buildPyramids() {
	buildLaplacianPyramid(left,leftLapPyr,leftHighestLevel);
	buildLaplacianPyramid(right,rightLapPyr,rightHighestLevel);
	buildGaussianPyramid();
    }
    
    
    void buildGaussianPyramid(){//金字塔内容为每一层的掩模
	assert(leftLapPyr.size()>0);
	
	maskGaussianPyramid.clear();
	Mat currentImg;
	cvtColor(blendMask,currentImg, CV_GRAY2BGR);//store color img of blend mask
	maskGaussianPyramid.push_back(currentImg); //0-level
	
	currentImg = blendMask;
	for (int l=1; l<levels+1; l++){
	    Mat _down;
	    if(leftLapPyr.size() > l)
		pyrDown(currentImg, _down, leftLapPyr[l].size());
	    else
		pyrDown(currentImg, _down, leftHighestLevel.size()); //lowest level
		
		Mat down;
	    cvtColor(_down, down, CV_GRAY2BGR);
	    maskGaussianPyramid.push_back(down);//add color blend mask into mask Pyramid
	    currentImg = _down;
	}
    }
    
    void buildLaplacianPyramid(const Mat& img, vector<Mat_<Vec3f> >& lapPyr, Mat& HighestLevel){
	lapPyr.clear();
	Mat currentImg = img;
	for (int l=0; l<levels; l++) {
	    Mat down,up;
	    pyrDown(currentImg, down);
	    pyrUp(down, up,currentImg.size());
	    Mat lap = currentImg - up;
	    lapPyr.push_back(lap);
	    currentImg = down;
	}
	currentImg.copyTo(HighestLevel);
    }
    
    Mat_<Vec3f> reconstructImgFromLapPyramid(){
	//将左右laplacian图像拼成的resultLapPyr金字塔中每一层
	//从上到下插值放大并相加,即得blend图像结果
	Mat currentImg = resultHighestLevel;
	for(int l=levels-1; l>=0; l--){
	    Mat up;
	    
	    pyrUp(currentImg, up, resultLapPyr[l].size());
	    currentImg = up + resultLapPyr[l];
	}
	return currentImg;
    }
    
    void blendLapPyrs(){
	//获得每层金字塔中直接用左右两图Laplacian变换拼成的图像resultLapPyr
	resultHighestLevel = leftHighestLevel.mul(maskGaussianPyramid.back()) + 
	rightHighestLevel.mul(Scalar(1.0,1.0,1.0) - maskGaussianPyramid.back());
	for (int l=0; l<levels; l++){
	    Mat A = leftLapPyr[l].mul(maskGaussianPyramid[l]);
	    Mat antiMask = Scalar(1.0, 1.0, 1.0) - maskGaussianPyramid[l];
	    Mat B = rightLapPyr[l].mul(antiMask);
	    Mat_<Vec3f> blendedLevel = A + B;
	    
	    resultLapPyr.push_back(blendedLevel);
	}
    }
    
public:
    LaplacianBlending(const Mat_<Vec3f>& _left, const Mat_<Vec3f>& _right, const Mat_<float>& _blendMask, int _levels)://construct function, used in LaplacianBlending lb(l,r,m,4);
    left(_left),right(_right),blendMask(_blendMask),levels(_levels)
    {
	assert(_left.size() == _right.size());
	assert(_left.size() == _blendMask.size());
	buildPyramids();  //construct Laplacian Pyramid and Gaussian Pyramid
	blendLapPyrs();   //blend left & right Pyramids into one Pyramid
    };
    
    Mat_<Vec3f> blend() {
	return reconstructImgFromLapPyramid();//reconstruct Image from Laplacian Pyramid
    }
};

Mat_<Vec3f> LaplacianBlend(const Mat_<Vec3f>& l, const Mat_<Vec3f>& r, const Mat_<float>& m) {
    LaplacianBlending lb(l,r,m,4);
    return lb.blend();
}




Rect get_overflap_region(vector<Point> &overflap_corners)
{
    int min_x,min_y,max_x,max_y,w,h;
    min_x=max_x=overflap_corners[0].x;
    min_y=max_y=overflap_corners[0].y;
    for(int i=1;i<overflap_corners.size();i++)
    {
	if(overflap_corners[i].x>max_x)
	    max_x=overflap_corners[i].x;
	if(overflap_corners[i].y>max_y)
	    max_y=overflap_corners[i].y;
	
	if(overflap_corners[i].x<min_x)
	    min_x=overflap_corners[i].x;
	if(overflap_corners[i].y<min_y)
	    min_y=overflap_corners[i].y;
	
    }
    
    w=max_x-min_x+2;
    h=max_y-min_y+2;
    min_x-=1;
    min_y-=1;
    return Rect(min_x,min_y,w,h);
    
}
void seamline(unsigned char* img, int width,int height,Mat& seam_mask)
{
    
    int j,min_j=width/2;
    //cout<<" min_j "<<min_j<<endl;
    int i=0;
    uchar min_cost;
    long index;
    seam_mask.at<uchar>(i,min_j)=255;
    seam_mask.at<uchar>(i,min_j-1)=255;
    seam_mask.at<uchar>(i,min_j+1)=255;
  
    for(int i=1;i<height;i++)
    {
	//cout<<i<<"         i   "<<width<<"  "<<height<<"  "<<min_j<<endl;
	index=i*width+min_j;
	min_cost=img[index];
	if(min_cost==0)
	{
	    for(int k=-1;k<=1;k++)
	    {
		j=min_j+k;
		if(j>0&&j<width)
		    seam_mask.at<uchar>(i,j)=255;
	    }
	    continue;
	    
	}
	for(int k=-1;k<=1;k++)
	{
	    j=min_j+k;
	    if(j<0||j>width-1)
		continue;
	    if(img[i*width+j]<min_cost)
	    {
		//cout<<img[i*width+j]-'0'<<" ";
		min_j=j;
		//cout<<"k   "<<k<<endl;
	    }
	    //cout<<endl;
	}
	for(int k=-1;k<=1;k++)
	{
	    j=min_j+k;
	    if(j>0&&j<width)
		seam_mask.at<uchar>(i,j)=255;
	}
    }
    for(int i=0;i<height;i++)
	for(int j=0;j<width;j++)
	{
	    if(seam_mask.at<uchar>(i,j)==255)
		break;
	    seam_mask.at<uchar>(i,j)=255;
	}
    return;
}


void seamline_fullimage(unsigned char* img, int width,int height,Mat& seam_mask,int &start_x, int & end_x)
{
    
    int j,min_j=width/2;
    start_x=min_j;
    //cout<<" min_j "<<min_j<<endl;
    int i=0;
    uchar min_cost;
    long index;
    seam_mask.at<uchar>(i,min_j)=255;
    seam_mask.at<uchar>(i,min_j-1)=255;
    seam_mask.at<uchar>(i,min_j+1)=255;
    
    
    for(int i=1;i<height;i++)
    {
	//cout<<i<<"         i   "<<width<<"  "<<height<<"  "<<min_j<<endl;
	index=i*width+min_j;
	min_cost=img[index];
	if(min_cost==0)
	{
	    for(int k=-1;k<=1;k++)
	    {
		j=min_j+k;
		if(j>0&&j<width)
		    seam_mask.at<uchar>(i,j)=255;
	    }
	    continue;
	    
	}
	for(int k=-1;k<=1;k++)
	{
	    j=min_j+k;
	    if(j<0||j>width-1)
		continue;
	    if(img[i*width+j]<min_cost)
	    {
		//cout<<img[i*width+j]-'0'<<" ";
		min_j=j;
		//cout<<"k   "<<k<<endl;
	    }
	    //cout<<endl;
	}
	end_x=min_j;
	for(int k=-1;k<=1;k++)
	{
	    j=min_j+k;
	    if(j>0&&j<width)
		seam_mask.at<uchar>(i,j)=255;
	}
    }
  
    for(int i=0;i<height;i++)
	for(int j=0;j<width;j++)
	{
	    if(seam_mask.at<uchar>(i,j)==255)
		break;
	    seam_mask.at<uchar>(i,j)=255;
	}
	return;
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
	 //cout<<objs[i]<<"   "<<scenes[i]<<endl;
     }
     
     bool isflap=PolygonClip(objs,scenes,overflap_corners);

     
     //for(auto cor:overflap_corners)
	// cout<<"cor "<<cor<<endl;
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
		 //cout<<"____ "<<Point(x,y)<<" i,j "<<i<<j<<endl;
	     }
	 }
     }
     //cout<<"inter poly.size "<<interPoly.size()<<endl;
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
     //ClockwiseSortPoints(interPoly);
     vector<Point> hull;
     convexHull(interPoly,hull,true);
     interPoly.clear();
     interPoly.assign(hull.begin(),hull.end());
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
 void Multiblending(Mat3b& l8u,Mat3b& r8u,Mat& blend_mask,Mat3b& result)
 {
     Mat_<Vec3f> l; l8u.convertTo(l,CV_32F,1.0/255.0);
     Mat_<Vec3f> r; r8u.convertTo(r,CV_32F,1.0/255.0);
     Mat_<float> mm;
     //=(Mat<float>)blend_mask;
     //Mat_<unsigned int> blend2= blend_mask;
     blend_mask.convertTo(mm,CV_32F,1.0/255.0); 
     
     Mat_<Vec3f> blend = LaplacianBlend(l, r, mm);
     blend.convertTo(result,CV_8UC3,255);
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
	     //cout<<"line cross point "<<p1<<p2<<q1<<q2<<x<<y<<endl;
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
 