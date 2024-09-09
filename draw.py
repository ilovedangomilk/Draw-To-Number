from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import ImageGrab
import numpy as np
import cv2

# Load the trained model
model = load_model('./models/digit_recognition_model.h5')

# Tkinter-based drawing app
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        
        self.canvas = tk.Canvas(self.root, width=600, height=600, bg='white')
        self.canvas.pack()

        # Bind mouse events to the canvas
        self.canvas.bind('<B1-Motion>', self.paint)

        # Button to predict the digit
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        # Button to clear the canvas
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)

    def clear_canvas(self):
        self.canvas.delete('all')

    def predict_digit(self):
        # Capture the canvas as an image
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        
        # Capture the canvas and preprocess the image
        image = ImageGrab.grab().crop((x, y, x1, y1))
        digit_image = self.preprocess_image(image)

        # Predict the digit using the model
        prediction = model.predict(digit_image)
        digit = np.argmax(prediction)
        print(f"Predicted Digit: {digit}")

    def preprocess_image(self, image):
        # Convert to grayscale
        image = image.convert('L')
        image = image.resize((28, 28))
        
        # Invert the colors
        image = np.array(image)
        image = cv2.bitwise_not(image)

        # Normalize and reshape the image
        image = image / 255.0
        image = image.reshape(1, 28, 28, 1)
        return image

# Run the app
root = tk.Tk()
app = App(root)
root.mainloop()
