from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Example settings
max_seq_length = 7  # Max length of digit sequence
num_digits = 10  # 10 digits (0-9)

# Generate some dummy data for illustration purposes (replace with your real data)
# Input: sequences of digits (e.g., [1, 2, 3])
X_train = np.random.randint(0, num_digits, (1000, max_seq_length))
y_train = np.random.randint(0, num_digits, (1000, max_seq_length))

# Pad sequences to ensure they have the same length
X_train = pad_sequences(X_train, maxlen=max_seq_length, padding='post')
y_train = pad_sequences(y_train, maxlen=max_seq_length, padding='post')

# One-hot encode the target
y_train = np.eye(num_digits)[y_train]

# Define the input sequence (variable-length)
input_seq = Input(shape=(max_seq_length,))

# Embedding layer to represent digits (optional but useful for larger vocab)
embedding = Embedding(input_dim=num_digits, output_dim=64)(input_seq)

# LSTM layer for sequence modeling
lstm_out = LSTM(128, return_sequences=True)(embedding)

# Output layer: predicting a sequence of digits
output = Dense(num_digits, activation='softmax')(lstm_out)

# Define the model
model = Model(input_seq, output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the model
model.save('./models/seq2seq_digit_model.h5')
