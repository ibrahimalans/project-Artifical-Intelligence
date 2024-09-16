
# Object Detection Project using OpenCV
## Student Names

- [Ibrahim Abdo Saleh Al-Anas] [202174312]
- [Mohammad ] []

## Overview

This project uses OpenCV, a popular computer vision library, to perform object detection on images or video streams. The object detection is achieved through the SSD (Single Shot MultiBox Detector) model with a MobileNetV3 backbone, which is a deep learning model for real-time object detection.

## Components of the Project

1. **Image Processing Function (`ImgFile()`)**:
   - **Purpose**: Detects objects in a static image and displays the results.
   - **Functionality**:
     - Reads an image file named `'person.png'`.
     - Loads the pre-trained SSD model configuration and weights.
     - Preprocesses the image and performs object detection.
     - Draws bounding boxes around detected objects and labels them.
     - Displays the annotated image.

2. **Video/Camera Processing Function (`Camera()`)**:
   - **Purpose**: Detects objects in real-time from a video feed or camera stream.
   - **Functionality**:
     - Captures video from the default camera (can be configured for an IP camera).
     - Loads the pre-trained SSD model configuration and weights.
     - Continuously reads frames from the video feed.
     - Performs object detection on each frame.
     - Draws bounding boxes around detected objects and labels them.
     - Displays the processed video feed in real-time.

## Detailed Explanation

### 1. Image Processing (`ImgFile()` function)

- **Image Loading**:
  ```python
  img = cv2.imread('person.png')
  ```
  This line loads the image file `'person.png'` into the variable `img`.

- **Class Names**:
  ```python
  with open(classFile, 'rt') as f:
      classNames = f.read().rstrip('\n').split('\n')
  ```
  This block reads class names from `'coco.arabic.names'` file into a list. These class names are used for labeling detected objects.

- **Model Configuration**:
  ```python
  net = cv2.dnn.DetectionModel(weightpath, configPath)
  net.setInputSize(320 , 230)
  net.setInputScale(1.0 / 127.5)
  net.setInputMean((127.5, 127.5, 127.5))
  net.setInputSwapRB(True)
  ```
  This part initializes the SSD model with the specified configuration and weights. The `setInputSize`, `setInputScale`, `setInputMean`, and `setInputSwapRB` methods are used to configure the preprocessing of the input image.

- **Object Detection**:
  ```python
  classIds, confs, bbox = net.detect(img, confThreshold=0.1)
  ```
  This line performs object detection on the image with a confidence threshold of 0.1.

- **Drawing Bounding Boxes**:
  ```python
  for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
      cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
      cv2.putText(img, classNames[classId-1], (box[0] + 10, box[1] + 20),
                  cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
  ```
  Detected objects are drawn with bounding boxes and labeled with their class names.

- **Displaying the Image**:
  ```python
  cv2.imshow('Output', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```
  Displays the annotated image in a window.

### 2. Video/Camera Processing (`Camera()` function)

- **Video Capture Initialization**:
  ```python
  cam = cv2.VideoCapture(0)
  cam.set(3, 740)
  cam.set(4, 580)
  ```
  This initializes video capture from the default camera. The `set` method adjusts the resolution of the video feed.

- **Class Names**:
  Same as in the `ImgFile()` function.

- **Model Configuration**:
  Similar to the `ImgFile()` function.

- **Real-Time Detection**:
  ```python
  while True:
      success, img = cam.read()
      classIds, confs, bbox = net.detect(img, confThreshold=0.5)
      ...
  ```
  Continuously captures frames from the camera, performs object detection, and updates the video feed with bounding boxes and labels.

- **Displaying the Video Feed**:
  ```python
  k = cv2.resize(img, (500, 500))
  cv2.imshow('Output', k)
  cv2.waitKey(1)
  ```
  Displays the processed video feed with detected objects.

## Important Points

- **Model Files**:
  - `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` (Model configuration)
  - `frozen_inference_graph.pb` (Model weights)
  - `coco.arabic.names` or `coco.names` (Class names)

- **Thresholds**:
  - Confidence threshold for detection (`confThreshold`) determines the sensitivity of object detection. Lower values detect more objects, but may include false positives.

- **Error Handling**:
  - The `cv2.destroyAllwindows()` function should be corrected to `cv2.destroyAllWindows()` to ensure proper cleanup of OpenCV windows.

- **Library Requirements**:
  - This script requires the OpenCV library (`cv2`). Make sure you have it installed with the `opencv-python` package.


