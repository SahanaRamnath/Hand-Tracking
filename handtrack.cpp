//do meanshift program for the backprojected image
//then detect the red rectangle around the hand
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <highgui.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <opencv2/video/video.hpp>
using namespace cv;
using namespace std;

Mat srcimg,hsvimg,hueimg,dstimg;
int bins=25; //number of partitions of the range 0-255 ig
Mat mask,hsvimgnew,lower_red_hue_range,upper_red_hue_range,contourimgnew;     

Rect window(Point(139,193),Point(280,360));
vector<vector<Point> > contour;
vector<Vec4i> hierarchy;
//Scalar colour=Scalar(55,0,175);
Scalar colour=Scalar(249,0,249);
void histandbackproj(int,void*);
void performfunction();

int main(int argc, char** argv)
{
     VideoCapture vdo(0);
     cout<<"Once you read this message, press Enter. A window will open, and you have to place your hand inside the rectangle drawn. Then, press 's' ";
     getchar();
     while((char)waitKey(30)!='s')
     {bool frameread=vdo.read(srcimg);
     //imshow("Source",srcimg);
     rectangle(srcimg,window,Scalar(0,0,255),1,8);
    //rectangle(srcimg,window,colour,1,8);
     imshow("Source",srcimg);
     }
     destroyAllWindows();
     while(1)
     {
     bool srcread=vdo.read(srcimg);     
     //rectangle(srcimg,window,Scalar(0,255,0),1,8);
     //imshow( "SourceImage", srcimg );
     //if((char)waitKey(30)=='s')
     
     cvtColor( srcimg, hsvimg, CV_BGR2HSV );
     
     //Extract hue part of HSV Image
     hueimg.create( hsvimg.size(), hsvimg.depth() );
     int ch[]={ 0, 0 };//to copy 0th channel of source image to 0th channel of                           destination image//called index pairs
     int numberofindexpairs=1,numberofsources=1,numberofdests=1;
     mixChannels( &hsvimg, numberofsources, &hueimg, numberofdests, ch,                          numberofindexpairs );
 
     //creating trackbar to enter number of bins 
     //Doubt : Didn't we already declare it?
     namedWindow( "SourceImage", CV_WINDOW_AUTOSIZE );
     int maxsliderposition=180;
     createTrackbar( "Hue", "SourceImage", &bins, maxsliderposition,                                                   histandbackproj );
     histandbackproj( 0, 0 );
     
     
    // imshow( "SourceImage", srcimg );
     if(waitKey(30)==27) break;
     }
     
     waitKey(0);
     return -1;
}

void histandbackproj(int, void* )
{
     Mat hist;
     int histsize=MAX( bins, 2 );
     float huerange[]={0,180};//see calcHist prototype for explanation
     const float* range={huerange};
     
     //Get histogram for the hue channel and normalize it
     calcHist( &hueimg, 1, 0, Mat(), hist, 1, &histsize, &range, true, false );
     normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );//Du

     //Getting Backprojection
     Mat backprojimg;
     calcBackProject( &hueimg, 1, 0, hist, backprojimg, &range, 1, true );
     //imshow("BackProjectionImage",backprojimg);
 
     //Drawing histogram
     int width=400,height=400;
     int bin_w=cvRound( (double) width /histsize );
     Mat histimg=Mat::zeros( width, height, CV_8UC3 );//wats hist then??
     for( int i=0; i<bins; i++ )
     {
          rectangle( histimg, Point(i*bin_w,height), 
                              Point((i+1)*bin_w,height-cvRound(hist.at<float>(i)*height/255.0)), Scalar(0,0,255), -1);
     }

     //imshow("Histogram", histimg );
     //imshow("HueImage", hueimg );

     //rectangle(srcimg,Point(50,90),Point(40,25),Scalar(0,0,255),1,8);     
     TermCriteria criteria(TermCriteria::EPS, 100, 0.00000000001 );

     //rectangle(srcimg,mean_shift,Scalar(0,255,0));
     //Rect window(250,90,300,100);
     meanShift(backprojimg, window, criteria );
     //CamShift(backprojimg, window, criteria );
     rectangle(srcimg,window,Scalar(0,0,255),1,8);
     cvtColor( srcimg, hsvimgnew, CV_BGR2HSV );
     /*for(z=1;z<31;z+=2)
     { GaussianBlur(hsvimgnew,hsvimgnew,Size(z,z),0,0);}*/ 
     //Not using GausianBlur because it slows down the program and doesn't give great results     
     inRange(hsvimgnew,Scalar(0,100,100),Scalar(10,255,255),lower_red_hue_range);
     inRange(hsvimgnew,Scalar(160,100,100),Scalar(179,255,255),upper_red_hue_range);
/*     inRange(hsvimgnew,Scalar(120,100,50.2),Scalar(120,100,100),lower_red_hue_range);
     inRange(hsvimgnew,Scalar(100,180,100),Scalar(255,255,255),upper_red_hue_range);
    */ mask=lower_red_hue_range|upper_red_hue_range;
     performfunction();
     //imshow("BackProjectionImage",backprojimg);
     //imshow( "SourceImage", srcimg );
     imshow("New",srcimg);
     imshow("Mask",mask);

}
void performfunction()
{
     contourimgnew=Mat::zeros(mask.size(),CV_8UC3);
     findContours(mask,contour,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
//     drawContours(contourimgnew,contour,-1,Scalar(255,255,255),2,8,hierarchy,2,Point());          
     Mat drawing=Mat::zeros(mask.size(),CV_8UC3);
     vector<Point> approxrect;
     for(size_t i=0;i<contour.size();i++) 
     {
          approxPolyDP(contour[i],approxrect,arcLength(Mat(contour[i]),true)*0.05,true);
          double area=contourArea(contour[i],false); cout<<area<<endl;
          if((approxrect.size()==4)&&(area>=16000)&&(area<24000))
          {
           drawContours(drawing,contour,i,Scalar(0,255,255),2,8,hierarchy,2,Point());
           vector<Point>::iterator vertex;
           for(vertex=approxrect.begin();vertex!=approxrect.end();vertex++)
            circle(drawing,*vertex,3,Scalar(0,0,255),1);
          }
     }
     namedWindow("Quads",CV_WINDOW_AUTOSIZE);
     imshow("Quads",drawing);
     //imshow("ContourImage",contourimgnew);
}
 
