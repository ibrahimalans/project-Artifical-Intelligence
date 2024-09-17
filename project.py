import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import tkinter as tk
from threading import Thread, Event

# Variables to control camera stopping
stop_event = Event()  # This event is used to signal when to stop the detection
detection_thread = None  # Variable to store the detection thread
dual_camera_thread = None  # Variable to store the dual camera thread

####### Object Detection Function for Single Camera #######
def detect_objects(language):
    global detection_thread
    stop_event.clear()  # Reset the stop event

    # Load the appropriate class file based on the selected language
    classFile = 'coco_arabic.names' if language == 'Arabic' else 'coco.names'

    with open(classFile, 'rt', encoding='utf-8') as f:
        classNames = f.read().rstrip('\n').split('\n')  # Read class names from the file

    # Load the model configuration and weights
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightpath = 'frozen_inference_graph.pb'

    # Initialize the object detection model
    net = cv2.dnn.DetectionModel(weightpath, configPath)
    net.setInputSize(320, 230)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean([127.5, 127.5, 127.5])
    net.setInputSwapRB(True)

    # Start capturing video from the selected camera
    cam = cv2.VideoCapture(1) if camera_selection.get() == "External" else cv2.VideoCapture(2)

    while not stop_event.is_set():  # Check if the stop event is set
        success, img = cam.read()  # Read a frame from the camera
        if not success:
            break  # Exit the loop if the frame is not read successfully

        classIds, confs, bbox = net.detect(img, confThreshold=0.5)  # Perform object detection

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                # Draw rectangle around detected objects
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

                # If Arabic language is selected, draw Arabic text
                if language == 'Arabic':
                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    font_path = "arial.ttf"  # Ensure you have this font
                    font = ImageFont.truetype(font_path, 20)

                    # Prepare Arabic text
                    text = classNames[classId - 1]
                    reshaped_text = arabic_reshaper.reshape(text)
                    bidi_text = get_display(reshaped_text)

                    # Draw the Arabic text on the image
                    draw.text((box[0] + 10, box[1] + 10), bidi_text, font=font, fill=(255, 255, 255))
                    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # Convert back to OpenCV format
                else:
                    # Draw English text directly on the image
                    cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

        cv2.imshow('Output', img)  # Display the processed image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit if 'q' is pressed

    cam.release()  # Release the camera resource
    cv2.destroyAllWindows()  # Close all OpenCV windows

####### Object Detection Function for Dual Cameras #######
def detect_objects_dual_camera(language):
    global dual_camera_thread
    stop_event.clear()  # Reset the stop event

    # Load the appropriate class file based on the selected language
    classFile = 'coco_arabic.names' if language == 'Arabic' else 'coco.names'

    with open(classFile, 'rt', encoding='utf-8') as f:
        classNames = f.read().rstrip('\n').split('\n')  # Read class names from the file

    # Load the model configuration and weights
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightpath = 'frozen_inference_graph.pb'

    # Initialize the object detection model
    net = cv2.dnn.DetectionModel(weightpath, configPath)
    net.setInputSize(320, 230)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean([127.5, 127.5, 127.5])
    net.setInputSwapRB(True)

    # Start capturing video from both cameras
    cam1 = cv2.VideoCapture(1)
    cam2 = cv2.VideoCapture(2)

    while not stop_event.is_set():  # Check if the stop event is set
        success1, img1 = cam1.read()  # Read a frame from the first camera
        success2, img2 = cam2.read()  # Read a frame from the second camera
        if not success1 or not success2:
            break  # Exit the loop if either frame is not read successfully

        classIds1, confs1, bbox1 = net.detect(img1, confThreshold=0.5)  # Perform object detection for the first camera
        classIds2, confs2, bbox2 = net.detect(img2, confThreshold=0.5)  # Perform object detection for the second camera

        # Process first camera
        if len(classIds1) != 0:
            for classId, confidence, box in zip(classIds1.flatten(), confs1.flatten(), bbox1):
                cv2.rectangle(img1, box, color=(0, 255, 0), thickness=2)
                if language == 'Arabic':
                    img_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    font = ImageFont.truetype("arial.ttf", 20)
                    text = classNames[classId - 1]
                    bidi_text = get_display(arabic_reshaper.reshape(text))
                    draw.text((box[0] + 10, box[1] + 10), bidi_text, font=font, fill=(255, 255, 255))
                    img1 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                else:
                    cv2.putText(img1, classNames[classId - 1], (box[0] + 10, box[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

        # Process second camera
        if len(classIds2) != 0:
            for classId, confidence, box in zip(classIds2.flatten(), confs2.flatten(), bbox2):
                cv2.rectangle(img2, box, color=(0, 255, 0), thickness=2)
                if language == 'Arabic':
                    img_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    font = ImageFont.truetype("arial.ttf", 20)
                    text = classNames[classId - 1]
                    bidi_text = get_display(arabic_reshaper.reshape(text))
                    draw.text((box[0] + 10, box[1] + 10), bidi_text, font=font, fill=(255, 255, 255))
                    img2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                else:
                    cv2.putText(img2, classNames[classId - 1], (box[0] + 10, box[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

        # Concatenate both camera feeds horizontally
        combined_img = np.hstack((img1, img2))

        cv2.imshow('Dual Camera Output', combined_img)  # Display the combined image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit if 'q' is pressed

    cam1.release()  # Release both cameras
    cam2.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows

####### Control Functions #######
def stop_current_cameras():
    stop_event.set()  # Set the stop event to stop the current detection thread
    if detection_thread and detection_thread.is_alive():  # If the detection thread is running
        detection_thread.join()  # Wait for the detection thread to finish
    if dual_camera_thread and dual_camera_thread.is_alive():  # If the dual camera thread is running
        dual_camera_thread.join()  # Wait for the dual camera thread to finish
    cv2.destroyAllWindows()  # Close all OpenCV windows

def start_detection():
    global detection_thread
    stop_current_cameras()  # Stop any running camera detection first
    stop_event.clear()  # Reset the stop event for the new detection
    detection_thread = Thread(target=detect_objects, args=(language_selection.get(),))  # Start the detection thread
    detection_thread.start()  # Start the thread

def start_dual_camera_detection():
    global dual_camera_thread
    stop_current_cameras()  # Stop any running camera detection first
    stop_event.clear()  # Reset the stop event for the new detection
    dual_camera_thread = Thread(target=detect_objects_dual_camera, args=(language_selection.get(),))  # Start the dual camera detection thread
    dual_camera_thread.start()  # Start the thread

def toggle_camera_mode():
    if camera_selection.get() == "Dual":
        start_dual_camera_detection()  # Start dual camera detection
    else:
        start_detection()  # Start single camera detection

####### User Interface Setup #######
root = tk.Tk()
root.title("Object Detection")

# Set window dimensions (10cm x 10cm)
width = 400  # Approximately 10cm in pixels
height = 400  # Approximately 10cm in pixels
root.geometry(f"{width}x{height}")

# Message about the engineers
message = "Engineers:\nIbrahim Al-Anas\nMohammed Al-Bouani"
tk.Label(root, text=message, font=("Arial", 16), justify="center").pack(pady=10)

# Language selection
tk.Label(root, text="Choose Language:").pack()
language_selection = tk.StringVar(value="English")
tk.Radiobutton(root, text="English", variable=language_selection, value="English").pack()
tk.Radiobutton(root, text="Arabic", variable=language_selection, value="Arabic").pack()

# Camera selection
tk.Label(root, text="Choose Camera:").pack()
camera_selection = tk.StringVar(value="Internal")
tk.Radiobutton(root, text="External Camera", variable=camera_selection, value="Internal").pack()
tk.Radiobutton(root, text="Internal Camera", variable=camera_selection, value="External").pack()
tk.Radiobutton(root, text="Dual Cameras", variable=camera_selection, value="Dual").pack()
# Control buttons
start_button = tk.Button(root, text="Start Detection", command=toggle_camera_mode)
start_button.pack(pady=5)

stop_button = tk.Button(root, text="Stop Detection", command=stop_current_cameras)
stop_button.pack(pady=5)


root.mainloop()  # Start the Tkinter event loop