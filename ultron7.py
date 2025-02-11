import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import os
import numpy as np

# Initialize webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create necessary folders
image_folder = "captured_images"
video_folder = "captured_videos"
face_detect_folder = "face_detected_images"
os.makedirs(image_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)
os.makedirs(face_detect_folder, exist_ok=True)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize Tkinter window
root = tk.Tk()
root.title("Thermal Vision Camera with Face Detection and Color Legend")

# Create video label
video_label = Label(root)
video_label.pack()

# Initialize counters and recording state
image_counter = 1
video_counter = 1
recording = False
video_writer = None

# Function to apply custom thermal filter
def apply_custom_thermal_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    thermal_frame = cv2.applyColorMap(enhanced_gray, cv2.COLORMAP_JET)
    return thermal_frame

# Function to add thermal color legend to the image
def add_thermal_color_legend(image):
    legend_text = """
    Color Codes for Thermal Image:
    Blue - Cool Areas
    Cyan - Mild Heat
    Green - Moderate Heat
    Yellow - Warm Areas
    Orange - Hot Areas
    Red - Very Hot Areas
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "Thermal Color Code Legend", (20, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, legend_text, (20, 60), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image

# Function to detect and highlight face in the thermal image
def detect_face_in_thermal_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red rectangle around the face

        detected_path = os.path.join(face_detect_folder, f"face_detected_{image_counter}.png")
        image = add_thermal_color_legend(image)  # Add the thermal color legend to the image
        cv2.imwrite(detected_path, image)
        print(f"Face detected and saved as '{detected_path}'.")
    else:
        print("No face detected.")

# Function to capture image
def capture_image():
    global image_counter
    filename = os.path.join(image_folder, f"captured_image_{image_counter}.png")
    cv2.imwrite(filename, thermal_frame)
    print(f"Image captured and saved as '{filename}'.")
    detect_face_in_thermal_image(filename)
    image_counter += 1

# Function to toggle recording
def toggle_recording():
    global recording, video_writer, video_counter
    if recording:
        recording = False
        video_writer.release()
        video_writer = None
        record_button.config(text="Start Recording", bg="green")
        print("Recording stopped.")
    else:
        recording = True
        filename = os.path.join(video_folder, f"thermal_video_{video_counter}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(webcam.get(3))
        frame_height = int(webcam.get(4))
        video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))
        video_counter += 1
        record_button.config(text="Stop Recording", bg="red")
        print(f"Recording started: {filename}")

# Function to update the video frame
def update_frame():
    global thermal_frame
    grabbed, frame = webcam.read()
    if not grabbed:
        return
    thermal_frame = apply_custom_thermal_filter(frame)
    img = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.img_tk = img_tk
    video_label.configure(image=img_tk)
    if recording and video_writer:
        video_writer.write(thermal_frame)
    root.after(10, update_frame)

# Create buttons
capture_button = Button(root, text="Capture", command=capture_image, font=("Arial", 14), bg="blue", fg="white")
capture_button.pack(pady=5)

record_button = Button(root, text="Start Recording", command=toggle_recording, font=("Arial", 14), bg="green", fg="white")
record_button.pack(pady=5)

# Start updating the video frame
update_frame()
root.mainloop()

# Release resources
webcam.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
