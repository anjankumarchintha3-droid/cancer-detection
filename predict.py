import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("cancer_detection_model.h5")
print("✅ Model loaded successfully!")

# Get new patient data
print("\n--- Cancer Prediction System ---")
new_data = input("Enter patient test data (comma-separated values): ")

# Convert input to numpy array
new_data = np.array(new_data.split(','), dtype=float)
new_data = new_data.reshape(1, -1)

# Predict
prediction = model.predict(new_data)

# Show result
if prediction[0][0] > 0.5:
    print("🩸 Result: Cancer (Positive)")
else:
    print("✅ Result: No Cancer (Negative)")