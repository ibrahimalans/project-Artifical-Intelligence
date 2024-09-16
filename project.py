import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import tkinter as tk
from threading import Thread, Event

# Variable to control camera stopping
stop_event = Event()  # This event is used to signal when to stop the detection
detection_thread = None  # Variable to store the detection thread

####### Object Detection Function #######
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
    cam = cv2.VideoCapture(0) if camera_selection.get() == "Internal" else cv2.VideoCapture(1)

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

####### Control Functions #######
def start_detection():
    global detection_thread
    stop_event.clear()  # Reset the stop event
    detection_thread = Thread(target=detect_objects, args=(language_selection.get(),))  # Start the detection thread
    detection_thread.start()  # Start the thread

def stop_detection():
    stop_event.set()  # Set the stop event
    if detection_thread:  # If the detection thread is running
        detection_thread.join()  # Wait for it to finish

def restart_program():
    """Restart the program"""
    python = os.executable
    os.execl(python, python, *os.sys.argv)  # Execute a new instance of the program

# User Interface Setup
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

# Control buttons
start_button = tk.Button(root, text="Start Detection", command=start_detection)
start_button.pack(pady=5)

stop_button = tk.Button(root, text="Stop Detection", command=stop_detection)
stop_button.pack(pady=5)

# Restart button
restart_button = tk.Button(root, text="Restart Program", command=restart_program)
restart_button.pack(pady=5)

root.mainloop()  # Start the Tkinter event loop