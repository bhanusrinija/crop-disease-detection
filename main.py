import tkinter as tk
from tkinter import filedialog, Label, Button, Text, Scrollbar, messagebox, Frame, Canvas
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import google.generativeai as genai
import re
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load API key from environment variable
API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Please set the GENAI_API_KEY environment variable.")

# Configure Gemini API
genai.configure(api_key=API_KEY)

class PlantDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Detector")
        self.root.geometry("800x1000")
        self.root.configure(bg='#f0f4f9')

        # Create main container frame
        self.main_container = Frame(self.root, bg='#f0f4f9')
        self.main_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # Create a canvas with scrollbar
        self.canvas = Canvas(self.main_container, bg='#f0f4f9')
        self.scrollbar = Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        
        # Create a frame inside the canvas
        self.scrollable_frame = Frame(self.canvas, bg='#f0f4f9')
        
        # Configure canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Center the content
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(1, weight=1)
        
        self.canvas.grid(row=0, column=1, sticky="nsew")
        self.scrollbar.grid(row=0, column=2, sticky="ns")

        # Load model
        self.model = load_model(r"C:\Users\DELL\Bhanu Srinija Projects\crop gpt\PlantVillage.h5")
        
        # Class labels
        self.class_labels = [
            'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 
            'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
            'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
            'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
            'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
        ]

        # Create UI
        self.create_widgets()

    def create_widgets(self):
        """Create the UI components for the application."""
        # Title
        title_label = Label(
            self.scrollable_frame, 
            text="Plant Disease Detector", 
            font=("Helvetica", 20, "bold"), 
            bg='#f0f4f9', 
            fg='#2c3e50'
        )
        title_label.pack(pady=(10, 20), anchor='center')

        # Image Display
        self.image_frame = Frame(self.scrollable_frame, bg='white', width=500, height=400)
        self.image_frame.pack(pady=20, anchor='center')
        self.image_frame.pack_propagate(False)

        self.image_label = Label(
            self.image_frame, 
            bg='white', 
            text="Upload or Capture an Image", 
            font=("Helvetica", 14)
        )
        self.image_label.pack(expand=True)

        # Buttons
        button_frame = Frame(self.scrollable_frame, bg='#f0f4f9')
        button_frame.pack(pady=20, anchor='center')

        # Stylish buttons
        camera_btn = Button(
            button_frame, 
            text="Open Camera", 
            command=self.open_camera,
            bg='#3498db', 
            fg='white', 
            font=("Helvetica", 10),
            padx=10,
            pady=8
        )
        camera_btn.pack(side=tk.LEFT, padx=15)

        upload_btn = Button(
            button_frame, 
            text="Upload Image", 
            command=self.upload_image,
            bg='#2ecc71', 
            fg='white', 
            font=("Helvetica", 10),
            padx=10,
            pady=8
        )
        upload_btn.pack(side=tk.LEFT, padx=15)

        # Result Label
        self.result_label = Label(
            self.scrollable_frame, 
            text="Prediction will appear here", 
            font=("Helvetica", 16), 
            bg='#f0f4f9',
            fg='#34495e'
        )
        self.result_label.pack(pady=20, anchor='center')

        # Disease Info Frame with increased height and scrollbar
        info_frame = Frame(self.scrollable_frame, bg='#f0f4f9')
        info_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True, anchor='center')

        # Scrollbar for text
        text_scrollbar = Scrollbar(info_frame)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,10))

        # Text widget with larger size and more lines
        self.info_text = Text(
            info_frame, 
            wrap=tk.WORD, 
            height=30,  # Increased height
            width=100,   # Increased width
            font=("Helvetica", 14),
            bg='white',
            fg='#2c3e50',
            padx=15,
            pady=15,
            yscrollcommand=text_scrollbar.set  
        )
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure text scrollbar
        text_scrollbar.config(command=self.info_text.yview)

    def predict_image(self, img_path):
        """Predict the disease from the uploaded image."""
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.
            
            class_probabilities = self.model.predict(img_array)
            predicted_label = np.argmax(class_probabilities)
            confidence = class_probabilities[0][predicted_label] * 100
            return self.class_labels[predicted_label], confidence
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return None, None

    def get_disease_info(self, disease_name):
        """Retrieve information about the disease from the Generative AI model."""
        try:
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"""Provide a concise, practical overview of {disease_name} for farmers:
            - Key symptoms
            - Quick identification methods
            - Basic prevention steps
            - Simple treatment recommendations"""

            response = model.generate_content(prompt)
            
            # Clean and format the response
            info = response.text
            info = re.sub(r'\*\*', '', info)  # Remove bold markers
            return info
        except Exception as e:
            return f"Could not retrieve disease information: {str(e)}"

    def open_camera(self):
        """Open the camera to capture an image."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Camera Error", "Failed to capture frame")
                break

            cv2.imshow("Capture Image (Press 's' to save, 'q' to quit)", frame)
            key = cv2.waitKey(1)
            
            if key == ord('s'):
                temp_path = "captured_image.jpg"
                cv2.imwrite(temp_path, frame)
                cap.release()
                cv2.destroyAllWindows()
                self.process_image(temp_path)
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

    def upload_image(self):
        """Upload an image from the file system."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )
        if file_path:
            self.process_image(file_path)

    def process_image(self, img_path):
        """Process the uploaded or captured image and display the prediction."""
        try:
            # Display image
            img = Image.open(img_path)
            img.thumbnail((200, 300))
            img_tk = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=img_tk, text="")
            self.image_label.image = img_tk

            # Predict disease
            prediction, confidence = self.predict_image(img_path)
            if prediction:
                # Update result label
                self.result_label.config(
                    text=f"Detected: {prediction}\nConfidence: {confidence:.2f}%"
                )

                # Get disease info
                disease_info = self.get_disease_info(prediction)
                
                # Clear and update info text
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, disease_info)

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = PlantDiseaseApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
