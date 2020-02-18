#include "cv.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace cv::ml;

int main(){
    VideoCapture cap("cv_live.mp4");
    if (!cap.isOpened()) {
        cout << "can't open video file" << endl;
        return 0;
    }
    Mat frame;
    vector<Point> track;

    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    int cur;
    int prev;
    int pprev;
    int ppprev;
    int count[6]={0};
    while(1){
        cap >> frame;
        if (frame.empty()) break;
        vector<Rect> found;
        vector<double> weights;
        Mat image = frame.clone();

        hog.detectMultiScale(image, found, weights);
        ppprev=pprev;
        pprev=prev;
        prev=cur;
        cur=found.size();

        for (size_t i = 0; i < (int)found.size(); i++){
            Rect rect = found[i];
            rectangle(frame, found[i], Scalar(0, 255, 0), 3);
            stringstream temp;
            temp << weights[i];
            putText(image, temp.str(),Point(found[i].x,found[i].y+50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255));
            track.push_back(Point(found[i].x+found[i].width/2,found[i].y+found[i].height/2));
        }

            for(size_t i = 1; i < track.size(); i++){
            line(image, track[i-1], track[i], Scalar(255,255,0), 2);
            }

        //enter
        if(cur==1&&prev==1&&pprev==1&&ppprev==0){
            count[0]=5;
        }
        count[0]--;
        if(count[0]<0) count[0]=0;
        if(count[0]>0){
            putText(frame, "Object 1 has entered", Point(frame.cols/4, frame.rows/2-70), 0, 1, Scalar(0, 0, 255), 3);
        }
        //cout << cur << " "<<prev<<" "<<pprev<<" "<<ppprev<<"->"<<count[0]<<endl;
        if(cur==2&&prev==2&&pprev==2&&ppprev==1){
            count[1]=5;
        }
        count[1]--;
        if(count[1]<0) count[1]=0;
        if(count[1]>0)
            putText(frame, "Object 2 has entered", Point(frame.cols/4, frame.rows/2-20), 0, 1, Scalar(0, 0, 255), 3);

        if(cur==3&&prev==3&&pprev==3&&ppprev==2){
            count[2]=5;
        }
        count[2]--;
        if(count[2]<0) count[2]=0;
        if(count[2]>0)
            putText(frame, "Object 3 has entered", Point(frame.cols/4, frame.rows/2+30), 0, 1, Scalar(0, 0, 255), 3);



        //walked out
        if(cur==0&&prev==1&&pprev==1&&ppprev==1){
            count[3]=5;
        }
        count[3]--;
        if(count[3]<0) count[3]=0;
        if(count[3]>0)
            putText(frame, "Object 2 has walked out", Point(frame.cols/4, frame.rows/2-70), 0, 1, Scalar(255, 0, 0), 3);

        if(cur==1&&prev==2&&pprev==2&&ppprev==2){
            count[4]=5;
        }
        count[4]--;
        if(count[4]<0) count[4]=0;
        if(count[4]>0)
            putText(frame, "Object 1 has walked out", Point(frame.cols/4, frame.rows/2-20), 0, 1, Scalar(255, 0, 0), 3);

        if(cur==2&&prev==3&&pprev==3&&ppprev==3){
            count[5]=5;
        }
        count[5]--;
        if(count[5]<0) count[5]=0;
        if(count[5]>0)
            putText(frame, "Object 3 has walked out", Point(frame.cols/4, frame.rows/2+30), 0, 1, Scalar(255, 0, 0), 3);
        

        char buff[100];
        if(cur==prev&&cur==pprev&&cur==ppprev)
        sprintf(buff, "There are a total number of %d objects in the room", cur);

        // sprintf(buff, "There are a total number of %lu objects in the room", found.size());
        putText(frame, buff, Point(0, frame.rows/8*7), 0, 1, Scalar(153, 102, 102), 3);
        imshow("21800412", frame);
        waitKey(33);
    }
    return 0;   
}