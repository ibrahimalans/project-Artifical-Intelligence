# Object Detection Project
## Student Names

- [Ibrahim Abdo Saleh Al-Anas] [202174312]
- [Mohammed A. AL-Bowani ] [202174103]

## Project Overview

This project is a real-time object detection application built using Python. The application detects objects in real-time from a live video stream (either from an internal or external camera) and displays bounding boxes around detected objects along with their names. The user can choose to display object names in either English or Arabic. The application uses a pre-trained deep learning model (SSD MobileNet) for object detection and offers a graphical user interface (GUI) built with Tkinter to control the detection process.

### Key Features:
- Real-time object detection from camera feed (internal or external).
- Choice of object labels in **English** or **Arabic**.
- A GUI with start, stop, and restart functionalities.
- Detection results displayed with bounding boxes and object names.
- Multithreading to ensure smooth execution of the object detection and GUI control.
- Ability to restart the program directly from the interface.

## Benefits of the Project
- **Real-Time Detection**: The application allows for real-time monitoring of objects using a webcam, which can be useful for security, automation, and various other applications.
- **Bilingual Interface**: The ability to display object names in both English and Arabic makes the application accessible to a wider audience.
- **Customizability**: Users can easily swap in other class files or models for different use cases, expanding its applicability.

## How the Project Works

1. **Object Detection Model**:
   - The project uses the **SSD MobileNet v3** deep learning model, pre-trained on the **COCO dataset**, which includes 91 common object categories (e.g., person, car, dog).
   - The model is loaded using OpenCV's DNN module, which performs the detection in real time on a live video stream.

2. **Language Selection**:
   - The application provides the ability to choose between **English** and **Arabic** for object labels.
   - Depending on the language selected, object names are displayed in the appropriate language on the video feed.

3. **Camera Selection**:
   - The user can choose between the **internal camera** (usually the built-in webcam on laptops) or an **external camera** if connected to the machine.
  
4. **Control Mechanisms**:
   - The application has a **Start Detection** button to initiate the object detection process in real-time.
   - The **Stop Detection** button stops the detection process and releases the camera.
   - A **Restart Program** button is available to restart the entire program, refreshing the interface and resetting the detection.

5. **GUI (Graphical User Interface)**:
   - The interface is built using **Tkinter**, which provides an intuitive way for users to interact with the application.
   - The interface is simple yet informative, displaying buttons to control the detection process and showing a message about the engineers who contributed to the project.

6. **Multithreading**:
   - The detection process runs in a separate thread to ensure the GUI remains responsive, even while object detection is actively running.
   - The detection can be stopped without freezing or crashing the application.

## How to Use the Project

### Prerequisites:
1. **Python 3.7+**
2. Install the required libraries by running the following command:
   ```bash
   pip install opencv-python-headless numpy pillow arabic-reshaper python-bidi
   ```
  
### Steps to Run the Project:
1. **Clone or Download** the project to your local machine.
2. **Run the application** by executing the Python script:
   ```bash
   python object_detection.py
   ```
3. The Tkinter GUI window will appear.
4. **Select the language** (English or Arabic) and **choose the camera** (internal or external) in the window.
5. Press the **Start Detection** button to begin detecting objects.
6. Press **Stop Detection** to stop the object detection and close the camera feed.
7. You can also press **Restart Program** to restart the entire application.

### Output:
- Once you start detection, a window will open displaying the camera feed.
- Detected objects will be highlighted with a green bounding box, and their names will be displayed either in **English** or **Arabic**, depending on the selected option.
- The program will display "Output" as the title of the window showing the detection results.

### Restarting:
- If you want to restart the application, simply click the "Restart Program" button in the interface, and it will reload.

### Stopping:
- The detection can be stopped by either pressing the **Stop Detection** button in the GUI or pressing the 'q' key in the output window.

## Important Files and Directories:
- **object_detection.py**: The main Python script that contains the code for object detection, GUI, and threading.
- **coco.names**: A text file containing object names in **English**.
- **coco_arabic.names**: A text file containing object names translated into **Arabic**.
- **ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt**: The configuration file for the SSD MobileNet model.
- **frozen_inference_graph.pb**: The pre-trained model used for object detection.

## Future Enhancements
1. **Model Customization**: Allow the user to upload their own models for detection, making it more versatile for custom tasks.
2. **Additional Languages**: Expand support to other languages.
3. **Save Results**: Add functionality to save detection results or output a log of detected objects.
4. **Improved GUI**: Further improve the GUI with more customization options or themes.

---

## Credits

Engineers:
- Ibrahim Al-Anas
- Mohammed Al-Bouani

This project was developed with the goal of creating a simple, bilingual object detection system that can be used for educational purposes or real-world applications.
