# train_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 1️⃣ Create sample data (4 features)
X, y = make_classification(
    n_samples=500,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# 2️⃣ Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Build a simple neural network model
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # sigmoid → binary output (0 or 1)
])

# 4️⃣ Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5️⃣ Train the model
model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

# 6️⃣ Evaluate accuracy
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# 7️⃣ Save the model
model.save('cancer_detection_model.h5')
print("✅ Model saved as 'cancer_detection_model.h5'")