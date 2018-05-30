//
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cv.h>
#include <highgui.h>
#include <RaspiCamCV.h>

#include "inc/cqt.h"
#include "inc/cqt_net.h"
#include "cqt_gen/cqt_gen.h"
#include "cqt_gen/cqt_debug.h"
#include "ya2k_yolo.h"

NUMPY_HEADER np;
#define IMG_SIZE 224 

float input_1_input [3][IMG_SIZE][IMG_SIZE];

int main(void)
{
  CQT_NET *tyolo_p;
  int ret;
  int is_person;
  YOLO_PARAM  yolo_parameter;

  CvScalar r_color = CV_RGB( 255,   0,   0 );
  CvScalar w_color = CV_RGB( 255, 255, 255 );
  CvPoint  str_pt, end_pt;
  CvFont   font;

  //-----------------------------
  // Initialize Camera Setting
  //-----------------------------
  IplImage *src_img = 0;
  IplImage *exe_img = 0;
  IplImage *dst_img = 0;

  RASPIVID_CONFIG * config = (RASPIVID_CONFIG*)malloc(sizeof(RASPIVID_CONFIG));
  config->width=640;
  config->height=480;
  config->bitrate=0;
  config->framerate=0;
  config->monochrome=0;
  RaspiCamCvCapture *video_cap = (RaspiCamCvCapture *) raspiCamCvCreateCameraCapture2(0, config);
  free(config);
  if (video_cap == NULL) {
    printf("[Error] : Camera Not Found\n");
    exit(1);
  }

  //-----------------------------
  // Create New Window
  //-----------------------------
  cvNamedWindow( "Tiny-YOLO Result (620x480)", CV_WINDOW_AUTOSIZE );
  cvMoveWindow ( "Tiny-YOLO Result (620x480)", 650, 50 );

  //-----------------------------
  // Initialize Cocytus 
  //-----------------------------
  tyolo_p = cqt_init();

  //-----------------------------
  // Load weight data 
  //-----------------------------
  ret = cqt_load_weight_from_files(tyolo_p, "weight/");
  if (ret != CQT_RET_OK) {
    printf("ERROR in cqt_load_weight_from_files %d\n", ret);
  }

#ifdef DEBUG
  printf("*** ESC-key to finish\n");
  while(1) {
#endif
      is_person = 0;
      //-----------------------------
      // Capture image data from camera
      //-----------------------------
      src_img = raspiCamCvQueryFrame(video_cap);

      exe_img = cvCreateImage(cvSize(224, 168), src_img->depth, src_img->nChannels);
      dst_img = cvCreateImage(cvSize(620, 480), src_img->depth, src_img->nChannels);
      cvResize(src_img, exe_img, CV_INTER_LINEAR);
      cvResize(src_img, dst_img, CV_INTER_LINEAR);

      //-----------------------------
      // Set image data to input layer 
      //-----------------------------
      for(int y=0;y<28;y++) {
        for(int x=0;x<IMG_SIZE;x++) {
          input_1_input[2][y][x] = ((float)128/256.0); 
          input_1_input[1][y][x] = ((float)128/256.0); 
          input_1_input[0][y][x] = ((float)128/256.0); 
        }
      } 
      for(int y=0;y<168;y++) {
        for(int x=0;x<IMG_SIZE;x++) {
          input_1_input[2][y+28][x] = ((float)CV_IMAGE_ELEM(exe_img, uchar, y, x*3+0)/256.0); 
          input_1_input[1][y+28][x] = ((float)CV_IMAGE_ELEM(exe_img, uchar, y, x*3+1)/256.0); 
          input_1_input[0][y+28][x] = ((float)CV_IMAGE_ELEM(exe_img, uchar, y, x*3+2)/256.0); 
        }
      } 
      for(int y=0;y<28;y++) {
        for(int x=0;x<IMG_SIZE;x++) {
          input_1_input[2][y+28+168][x] = ((float)128/256.0); 
          input_1_input[1][y+28+168][x] = ((float)128/256.0); 
          input_1_input[0][y+28+168][x] = ((float)128/256.0); 
        }
      } 

      //-----------------------------
      // Run Tiny-YOLO NN 
      //-----------------------------
      ret = cqt_run(tyolo_p, input_1_input);
      if(ret != CQT_RET_OK){
          printf("ERROR in cqt_run %d\n", ret);
      }

      //-----------------------------
      // Evaluate Tiny-YOLO NN Result 
      //-----------------------------
      yolo_parameter.width = 620;
      yolo_parameter.height = 480;
      yolo_parameter.score_threshold = 0.3;
      yolo_parameter.iou_threshold = 0.5;
      yolo_parameter.classes = 20;

      ret = yolo_eval(conv2d_9_output, &yolo_parameter);

      if(ret < 0) {
          printf("ERROR %d\n", ret);
          exit(1);
      }
      for(int i=0;i<ret;i++) {
          int class = yolo_result[i].class;
#ifdef DEBUG
          float score = yolo_result[i].score;

          BOX b = yolo_result[i].box;
          int top, left, bottom, right;

          top = (int)floor(b.top + 0.5);
          if(top < 0) {
              top = 0;
          }
          left = (int)floor(b.left + 0.5);
          if(left < 0) {
              left = 0;
          }
          bottom = (int)floor(b.bottom + 0.5);
          if(bottom >= yolo_parameter.height) {
              bottom = yolo_parameter.height - 1;
          }
          right = (int)floor(b.right + 0.5);
          if(right >= yolo_parameter.width) {
              right = yolo_parameter.width - 1;
          }

          printf("***Detect*** %s %d %f (%d, %d), (%d, %d)\n",
             voc_class[class], class, score, left, top, right, bottom);

          //-----------------------------
          // Frame for Text area
          //-----------------------------
          str_pt = cvPoint(  left    ,  top     );
          end_pt = cvPoint( (left+70), (top+15) );
          cvRectangle( dst_img, str_pt, end_pt, r_color, -1, 8, 0 );

          //-----------------------------
          // Frame for Detect area
          //-----------------------------
          str_pt = cvPoint( left , top    );
          end_pt = cvPoint( right, bottom );
          cvRectangle( dst_img, str_pt, end_pt, r_color, 1, 8, 0 );

          //-----------------------------
          // Text info.
          //-----------------------------
          str_pt = cvPoint( left, (top+15) );
          cvInitFont( &font, CV_FONT_HERSHEY_SIMPLEX, 0.6, 0.6, 0.0, 1, CV_AA );
          cvPutText( dst_img, voc_class[class], str_pt, &font, w_color );
#endif
          //-----------------------------
          // is Person
          //-----------------------------
	  if(class == 14) {
            is_person = 1;
//            printf("***Detect Person***\n");
	  }
      }

#ifdef DEBUG
    //-----------------------------
    // Show image
    //-----------------------------
//    cvShowImage("Camera (640x480)", src_img);
    cvShowImage("Tiny-YOLO Result (620x480)", dst_img);

    //-----------------------------
    // Timeout 1ms and check ESC-key
    //-----------------------------
    int key=cvWaitKey(500) & 0xFF;
    if (key==0x1b) {
        break;
    }
  }
#endif
  //-----------------------------
  // Closing process 
  //-----------------------------
//  cvDestroyWindow ("Tiny-YOLO Result (620x480)");
  raspiCamCvReleaseCapture( &video_cap );

  return is_person;
}

