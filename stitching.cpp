
 
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