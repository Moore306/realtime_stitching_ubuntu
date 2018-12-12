#include<video_stitching.h>
list<Point2f>  keypts1,keypts2;
Mat pre_frame1,pre_frame2;
cv::Mat color, depth, last_color;
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
    cout<<"test"<<endl;
    capture1.open("../imgs/test3.mkv");
    capture2.open("../imgs/test4.mkv");
    capture1.set(CV_CAP_PROP_FRAME_WIDTH, 1920);  
    capture1.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    //     //   
    capture2.set(CV_CAP_PROP_FRAME_WIDTH, 1920);  
    capture2.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    namedWindow("img11",0);
    namedWindow("img22",0);
    //namedWindow("result",0);
    namedWindow("show_region",0);
    vector<Point2f> kps1,kps2;
    char info[500];
    int w_index=0;
    unsigned char * img_p=nullptr;
    Mat3b result;
    int origin_w,origin_h;
    Mat show_img;
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
	//resize(img11,img11,Size(),0.3,0.3);
	//resize(img22,img22,Size(),0.3,0.3);
	origin_h=img11.rows;
	origin_w=img11.cols;
	cout<<origin_w<<"   "<<origin_h<<endl;
	//resize(img11,img11,Size(),0.3,0.3);
	//resize(img22,img22,Size(),0.3,0.3);
	resizeWindow("img11", 640, 480);
	resizeWindow("img22", 640, 480);
	//resizeWindow("result", 640, 480);
	resizeWindow("show_region",640,480);
	show_img=Mat::zeros(5*origin_h,5*origin_w,CV_8UC3);
	
	
	width=img11.cols;
	height=img11.rows;
	//continue;
	cvtColor(img11,img1,CV_BGR2GRAY);
	cvtColor(img22,img2,CV_BGR2GRAY);
	// 判断输入图像是否读取成功
	if (img1.empty() || img2.empty() || img1.channels() != 1 || img2.channels() != 1)
	{
	    cout << "Input Image is nullptr or the image channels is not gray!" << endl;
	    break;
	    //system("pause");
	}
	
	Mat1f H;
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	if(index++==0)
	{//orb for first frame 
	    
	    
	    regis(img1,img2,kps1,kps2,H);
	    cout<<"匹配点个数 "<<kps1.size()<<endl;
	    if(kps1.size()==0)
	    {
		index=0;
		continue;
	    }
	    
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
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t3-t1 );
	cout<<"stitching use time "<<time_used.count()<<" seconds."<<endl;
	
	Mat H1=Get_blendSize(width,height,H,offset,overflap_corners);
	
	//cout<<"get H1 "<<H1<<endl;
	H=(Mat1f)H1.clone();
	//cout<<"success "<<endl;
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
	Mat3b flapimg1,flapimg2; 
	Mat tiledImg;
	//Mat shftMat=(Mat_<double>(3,3)<<1.0,0,offset.x, 0,1.0,offset.y, 0,0,1.0);
	//warpPerspective(img11,tiledImg,shftMat*H,Size(width+50,height+50));
	
	Mat Mask(height,width,CV_8UC1),Mask2(height,width,CV_8UC1);
	Rect region=get_overflap_region(overflap_corners);
	Mask.setTo(0);Mask2.setTo(255);
	vector<vector<Point>> contours;
	contours.push_back(overflap_corners);
	fillPoly(Mask, contours,Scalar(255, 255, 255));
	warpPerspective(img11,tiledImg,H,Size(width,height));
	flapimg2=tiledImg(region).clone();
	tiledImg.setTo(0,Mask);
	img22.copyTo(tiledImg(Rect(abs(offset.x),abs(offset.y),img2.cols,img2.rows)));
	flapimg1=tiledImg(region).clone();
	tiledImg.setTo(0,Mask);
	Mat3b flap=flapimg1-flapimg2; //de dao chongdie quyu cha zhi
	int region_w=region.width,region_h=region.height;
	Mat seam_mask(region_h,region_w,CV_8UC1);
	seam_mask.setTo(0);
	seam_mask.setTo(0);
	img_p=flap.data+region_h*region_w;
	seamline(img_p, region_w,region_h,seam_mask);
	chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
	time_used = chrono::duration_cast<chrono::duration<double>>( t4-t3 );
	cout<<"flapover region  use time："<<time_used.count()<<" seconds."<<endl;
	sprintf(info,"Result/flap1_%02d.jpg",index);
	//imshow("seamline",seam_mask);
	//imwrite(info,flapimg1);
	sprintf(info,"Result/flap2_%02d.jpg",index);
	//imwrite(info,flapimg2);
	sprintf(info,"Result/flap_mask_%02d.jpg",index);
	//imwrite(info,seam_mask);
	//void Multiblending(Mat3b& l8u,Mat3b& r8u,Mat& blend_mask,Mat3b& result)
	//cout<<"result   __________________"<<endl;
	Multiblending(flapimg1,flapimg2,seam_mask,result);
	
	result.copyTo(tiledImg(region));
	sprintf(info,"Result/blending_result_%02d.jpg",index);
	//imshow("result",tiledImg);
	tiledImg.copyTo(Mat(show_img,Rect(0,region_h,tiledImg.cols,tiledImg.rows)));
	Mat show_region=show_img(Rect(abs(offset.x),abs(offset.y),2*img2.cols,2.5*img2.rows));
	imshow("show_region",show_region);
	//imwrite(info,tiledImg);
	t4 = chrono::steady_clock::now();
	time_used = chrono::duration_cast<chrono::duration<double>>( t4-t1 );
	cout<<"total  use time："<<time_used.count()<<" seconds."<<endl;

	
    }
}
