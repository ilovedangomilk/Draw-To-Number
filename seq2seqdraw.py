import tkinter as tk
from PIL import ImageGrab
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained sequence prediction model
model = load_model('./models/seq2seq_digit_model.h5')

# Define some constants
max_seq_length = 7  # Max number of digits that can be drawn
num_digits = 10  # 10 digits (0-9)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Sequence of Digits")

        # Create a canvas to draw digits
        self.canvas = tk.Canvas(self.root, width=400, height=200, bg='white')  # Space for 7 digits
        self.canvas.pack()

        # Bind mouse events to the canvas for drawing
        self.canvas.bind('<B1-Motion>', self.paint)

        # Create a button to predict the digit sequence
        self.predict_button = tk.Button(self.root, text="Predict Sequence", command=self.predict_sequence)
        self.predict_button.pack()

        # Create a button to clear the canvas
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

    def paint(self, event):
        """Draw on the canvas."""
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)

    def clear_canvas(self):
        """Clear the canvas."""
        self.canvas.delete('all')

    def predict_sequence(self):
        """Capture the canvas, preprocess the images, and predict the sequence of digits."""
        # Capture the canvas as an image (280x40 pixels)
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        # Capture the drawing as an image
        image = ImageGrab.grab().crop((x, y, x1, y1))

        # Convert the image to grayscale and split it into 7 segments (one for each digit)
        digit_images = self.preprocess_image(image)

        # Predict the digit sequence
        digit_sequence = self.predict_digit_sequence(digit_images)

        # Output the predicted sequence
        print(f"Predicted Digit Sequence: {digit_sequence}")

    def preprocess_image(self, image):
        """Preprocess the canvas image and split it into separate digit images."""
        # Convert the image to grayscale
        image = image.convert('L')

        # Resize the image to (280, 40) which contains 7 digits, each of 40x40 pixels
        image = image.resize((280, 40))

        # Convert to numpy array
        image = np.array(image)

        # Threshold the image (convert to binary black/white)
        _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

        # Split the image into 7 smaller images, each representing a digit (40x40 pixels)
        digit_images = []
        for i in range(max_seq_length):
            digit = image[:, i*40:(i+1)*40]  # Extract each 40x40 segment
            digit = cv2.resize(digit, (28, 28))  # Resize to 28x28 (like MNIST)
            digit = digit / 255.0  # Normalize to [0, 1]
            digit_images.append(digit)

        return np.array(digit_images)

    def predict_digit_sequence(self, digit_images):
        """Use the model to predict the sequence of digits."""
        # Reshape each digit to match the model input (batch_size, 28, 28, 1)
        digit_images = digit_images.reshape((1, max_seq_length, 28, 28, 1))

        # Preprocess by converting the images into a sequence of digits (0-9)
        # For simplicity, here we assume the input image already consists of pre-drawn digits
        # In a real-world case, you'd have a digit classifier to classify individual digits
        dummy_sequence = np.random.randint(0, num_digits, (1, max_seq_length))  # Use random sequence for now
        padded_sequence = pad_sequences(dummy_sequence, maxlen=max_seq_length, padding='post')

        # Predict the digit sequence
        predictions = model.predict(padded_sequence)

        # Convert softmax outputs to digit predictions
        predicted_digits = np.argmax(predictions, axis=-1)
        return predicted_digits[0]

# Run the application
root = tk.Tk()
app = App(root)
root.mainloop()
