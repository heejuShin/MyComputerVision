#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "cv.hpp"
#pragma comment(lib, "winmm.lib")
#include <fstream>
#include <iostream>
#include <cstring>

using namespace std;
using namespace cv;
using namespace dnn;

string guide[13]={"============","HAPPY TUBE","Hi, there. This program will help you as a Beauty Youtuber","This program automatically analyzes cosmetics when you see them on a set screen.","This will make your YouTube editing more simple.","The program is a trial version and can be used only for 30 seconds.","After 30 seconds, the program ends.","If you want to use this program more, please contact 21800412@handong.edu","Enjoy! Be the greatest Youtuber","--Enter option--","1)Webcam version","2)Video version : sample video is automatically used.","============"};
string endMessage[8]={"============","HAPPY TUBE","Did you enjoy HAPPY TUBE?", "if you want to use full version of HAPPYTUBE","contact 21800412@handong.edu","Be the greatest Youtuber","Thank U! :D","============"};


int ProgramGuide(Mat logo, Mat logo_gray, int option){
    //Mat image(700,1200, CV_8UC1, Scalar(100,100,100));
    Mat image=imread("background.png");
    resize(image, image, Size(1200,700), 0,0, CV_INTER_NN);
    int y=50;
    if (option==0){
        for(int i=0; i<14; i++){
            Point location(50,y);
            putText(image,guide[i],location,FONT_ITALIC,0.8,Scalar(0),2);
            y+=50;
        }
        Mat imageROI(image, Rect(image.cols-logo.cols, 0, logo.cols, logo.rows));
        Mat mask(200-logo_gray);
        logo.copyTo(imageROI, mask);
        imshow("Welcome to HAPPY_TUBE",image); 
        int key;
            while(1){
                key=waitKey();
                if(key == 49 || key == 50)  break;
            }
        waitKey(0);
        return key;
    }
    else if (option==1){
        for(int i=0; i<8; i++){
            Point location(50,y);
            putText(image,endMessage[i],location,FONT_ITALIC,0.8,Scalar(0),2);
            y+=50;
        }
        Mat imageROI(image, Rect(image.cols-logo.cols, 0, logo.cols, logo.rows));
        Mat mask(200-logo_gray);
        logo.copyTo(imageROI, mask);
        imshow("trial version End!",image); 
        waitKey(10000);
    }
    return 0;
}
int main(){
    Mat logo = imread("./logo.png");
    resize(logo, logo, Size(150,150), 0,0, CV_INTER_NN);
    Mat logo_gray;
    cvtColor(logo, logo_gray, COLOR_BGR2GRAY);
    int option=ProgramGuide(logo, logo_gray, 0);
    //설명 GUI로 나타내기
    if(option!=49 && option!=50){
        //return ProgramGuide(logo, logo_gray, 2);
        cout << "There's no such option. Exit the program."<<endl;
        return 0;
    }
    else{
        Mat frame;
        VideoCapture cap(0);
        if (option == 49 ){
        }
        else{
            if(cap.open("video.mp4")== 0){//sample video
                cout << "file disappear...!" << endl;
                waitKey(0);
            }
        }
        String modelConfiguration = "yolov2.cfg"; 
        String modelBinary = "yolov2.weights";
        Net net = readNetFromDarknet(modelConfiguration, modelBinary);
        vector<String> classNamesVec;
        ifstream classNamesFile("./coco.names");

        if (classNamesFile.is_open()) {
        string className = "";
        while (std::getline(classNamesFile, className)) classNamesVec.push_back(className);
    }

    //timer
        for(int i=0; i<(3*30); i++){
            //waitKey(1000);
        //}
        //while(1){
            cap >> frame;
            resize(frame, frame, Size(800,600),0,0,CV_INTER_LINEAR);

            //object detection
            if (frame.channels() == 4) cvtColor(frame, frame, COLOR_BGRA2BGR);
            Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false); 
            net.setInput(inputBlob, "data"); //set the network input
            Mat detectionMat = net.forward("detection_out"); //compute output
            float confidenceThreshold = 0.24; //by default 0.24
    
            for (int i = 0; i < detectionMat.rows; i++) { 
                const int probability_index = 5;
                const int probability_size = detectionMat.cols - probability_index;
                float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
                size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
                
                //특정한 물체가 detection된 확률
                float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

                //For drawing
                if (confidence > confidenceThreshold) {
                        float x_center = detectionMat.at<float>(i, 0) * frame.cols; 
                        float y_center = detectionMat.at<float>(i, 1) * frame.rows; 
                        float width = detectionMat.at<float>(i, 2) * frame.cols; 
                        float height = detectionMat.at<float>(i, 3) * frame.rows;

                        Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2)); 
                        Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2)); 
                        Rect object(p1, p2);
                        Scalar object_roi_color(128, 128, 255);

                        rectangle(frame, object, object_roi_color);
                        String className = objectClass < classNamesVec.size() ? classNamesVec[objectClass] : cv::format("unknown(%d)", objectClass); 
                        //String label = format("%s: %.2f", className.c_str(), confidence);
                        String label = format("%s", className.c_str());
                        int baseLine = 0;

                        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                        rectangle(frame, Rect(p1, Size(labelSize.width, labelSize.height + baseLine)), object_roi_color, FILLED);
                        putText(frame, label, p1 + Point(0, labelSize.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
                }
            }

            //아래로 로고 넣는 거
            //cout << "image : "<<frame.rows<<"x"<<frame.cols<<endl;
            //cout << "logo : "<<logo.rows<<"x"<<logo.cols<<endl;
            Mat imageROI(frame, Rect(frame.cols-logo.cols, 0, logo.cols, logo.rows));
            Mat mask(200-logo_gray);
            logo.copyTo(imageROI, mask);
            char time[10];
            int j= i/3;
            if((30-j)>9)
                sprintf(time, "00:%d", 30-j-1);
            else
            {
                sprintf(time, "00:0%d", 30-j-1);
            }
            
            int color=0;
            putText(frame,"[Trial Version]",Point(50,50), FONT_ITALIC,0.8,Scalar(255,255,255), 2);
            putText(frame,time,Point(70,100),FONT_ITALIC,0.8,Scalar(255-color,255-color,255),2);
            color-=8;
            imshow("HAPPYTUBE", frame);
            waitKey(30);
        }
        return ProgramGuide(logo, logo_gray, 1);
    }
    return 0;
}
