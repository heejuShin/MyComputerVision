#include "cv.hpp"
#include <iostream>
#include <opencv2/dnn.hpp>
#include <fstream>

using namespace cv;
using namespace std;
using namespace dnn;

int mynum=0;
void drawLine(vector<Vec2f> lines, float x, float y, float angle_th1, float angle_th2, Mat &result, int * plus);

int main(int argc, char** argv)
{
    VideoCapture cap("Go_1.mp4");
    //VideoCapture cap("Stop_1.mp4"); 
    //VideoCapture cap("Line_1.mp4");
    //VideoCapture cap("Line_2.mp4");
    //VideoCapture cap("Pedestrian_1.mp4");
    //VideoCapture cap("Pedestrian_2.mp4");
    //VideoCapture cap("Pedestrian_3.mp4");
    int pluss=0;
    int * plus= &pluss;
    Mat frame_ld, result, edge;
    vector<Vec2f> lines;

    
    String modelConfiguration = "yolov2.cfg";
    String modelBinary = "yolov2.weights";
    Net net = readNetFromDarknet(modelConfiguration, modelBinary);
    
    Mat inputBlob, detectionMat;
    float confidenceThreshold = 0.24;
    vector<String> classNamesVec;
    ifstream classNamesFile("coco.names");
    
    Mat frame, frame_roi;
    Rect rect(200, 200, 280, cap.get(CV_CAP_PROP_FRAME_HEIGHT) - 250), max_object;
    
    int maxSize_prev = 0, frame_cnt = 0, frame_freq = 10;
    float sub_size = 0;
    
    if (classNamesFile.is_open()) {
        string className = "";
        while (std::getline(classNamesFile, className)) classNamesVec.push_back(className);
    }
    //pedestrian
    Mat frame_gray, sub;
    vector<Rect> found;
    int i;
    char ch;
    vector<Mat> Background_Queue;
    HOGDescriptor hog(
                      Size(48, 96),
                      Size(16, 16),
                      Size(8, 8),
                      Size(8, 8),
                      9);
    hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());
    //pedestrian
    
    Mat frameln;
    int same=0, larger=0, smaller=0;
    while (1)
    {
        cap >> frame;

        if (frame.empty()) break;
        if (frame.channels() == 4) cvtColor(frame, frame, COLOR_BGRA2BGR);
        
        frame_roi = frame(rect);
        
        if(frame_cnt % frame_freq == 0){
            inputBlob = blobFromImage(frame_roi, 1 / 255.F, Size(416, 416), Scalar(), true, false);
            net.setInput(inputBlob, "data");
            detectionMat = net.forward("detection_out");
        }
        
        int maxSize = 0;
        string buff;
        bool check=false;
        for (int i = 0; i < detectionMat.rows; i++) {
            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float* prob_array_ptr = &detectionMat.at<float>(i, probability_index);
            size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
            
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
            
            if (confidence > confidenceThreshold) {
                
                float x_center = detectionMat.at<float>(i, 0) * frame_roi.cols;
                float y_center = detectionMat.at<float>(i, 1) * frame_roi.rows;
                float width = detectionMat.at<float>(i, 2) * frame_roi.cols;
                float height = detectionMat.at<float>(i, 3) * frame_roi.rows;
                Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
                Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));
                Rect object(p1, p2);
                Scalar object_roi_color(0, 255, 0);
                
                Size s = object.size();
                int object_size = s.width * s.height;
                
                if (maxSize < object_size) {
                    maxSize = object_size;
                }
                
                //rectangle(frame_roi, object, object_roi_color);
                String className = objectClass < classNamesVec.size() ? classNamesVec[objectClass] : cv::format("unknown(%d)", objectClass);
                String label;
                //String label = format("%s: %.2f", className.c_str(), confidence);
                buff=strdup(className.c_str());
                if(buff.compare("person")==1)
                    check=true;
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                
                rectangle(frame_roi, Rect(p1, Size(labelSize.width, labelSize.height + baseLine)), object_roi_color, FILLED);
                putText(frame_roi, label, p1 + Point(0, labelSize.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
            }
        }//object detection
        
        if(frame_cnt % frame_freq == 0){
            sub_size = (maxSize - maxSize_prev) / float(maxSize);
        }

        //pedestrian
        Rect pedR(100,frame.rows/4, frame.cols-200, frame.rows/4*3);
        Mat pedM = frame(pedR);
        //imshow("test",pedM);
        hog.detectMultiScale(pedM,
                        found,
                        1.1,
                        Size(8, 8),
                        Size(32, 32),
                        1.05,
                        8);
        //for (i = 0; i < (int)found.size(); i++)
        //    rectangle(frame, found[i], Scalar(0, 255, 0), 2);
        if((int)found.size()>0 && check){
            putText(frame, "Warning!: Collision with the pedestrian", Point(frame.cols/4, frame.rows/2-70), 0, 1, Scalar(0, 0, 255), 3);
        }
        if (abs(sub_size) < 0.05){
            same++; larger=0; smaller=0;
        }
        else if (sub_size >0){
            larger++; same=0; smaller=0;
            if(maxSize > 20000 && larger>10){
                putText(frame, "Warning! : Collision with front car", Point(frame.cols/4, frame.rows/2-35), 0, 1, Scalar(255, 255, 0), 3);           
            }
        }
        else{
            smaller++; larger=0; same=0;
            if(smaller>40 && maxSize < 12000 && maxSize >6000)
                putText(frame, "Warning!: Front car departure", Point(frame.cols/4, frame.rows/2+35), 0, 1, Scalar(0, 255, 0), 3);  
        }
        //printf("%d\n", maxSize);
        //printf("%d %d %d ->%d\n", smaller, same, larger, maxSize);
        
        maxSize_prev = maxSize;
        frame_cnt++;
        
        //line detection
        float x1=frame.cols/2-80;
        float y1=frame.rows-150;
        float wid = 160;
        float hei = 150;
        
        Rect ld(x1, y1, wid, hei);
        Mat frame_ld = frame(ld);
        //imshow("line", frame_ld);
        Canny(frame_ld, edge, 50, 150, 3);
        HoughLines(edge, lines, 1, CV_PI / 180, 50);
        drawLine(lines, x1, y1, -30, 30, frame, plus);

        imshow("21800412", frame);
        if (waitKey(33) >= 0) break;
    }
    return 0;
}


void drawLine(vector<Vec2f> lines, float x, float y, float angle_th1, float angle_th2, Mat &result, int * plus){
    float rho, theta, a, b, x0, y0;
    float avr_rho=0. , avr_theta=0.;
    int count = 0;
    Point p1, p2;
    
    for (int i = 0; i < lines.size(); i++) {
        rho = lines[i][0];
        theta = lines[i][1];
        
        if (theta < CV_PI / 180 * angle_th1 || theta > CV_PI / 180 * angle_th2) continue;
        
        avr_rho += rho;
        avr_theta += theta;
        count++;
    }
    avr_rho /= count;
    avr_theta /= count;
    a = cos(avr_theta);
    b = sin(avr_theta);
    x0 = a * avr_rho;
    y0 = b * avr_rho;
    
    p1 = Point(cvRound(x0 + 1000 * (-b)) + x, cvRound(y0 + 1000 * a) + y);
    p2 = Point(cvRound(x0 - 1000 * (-b)) + x, cvRound(y0 - 1000 * a) + y);
    
    line(result, p1, p2, Scalar(0, 255, 255), 3, 8);
    if((p1.x>0||p1.y>0)&&(p2.x>0||p2.y>0)){
        mynum=3;
    }
    if(mynum>0)
        putText(result, "Warning!: Lane departure", Point(result.cols/4, result.rows/2), 0, 1, Scalar(0, 255, 255), 3);
    mynum--;
}
