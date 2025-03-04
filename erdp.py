import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

# 1. Load the ECG Image
image_path = r'D:\Users\A.S.MUTHUVEL\Downloads\DAY 2\task2\Data\MI(1).jpg'  # Path to your ECG report image
img = cv2.imread(image_path)
if img is None:
    print("Error: Image not loaded. Check the image path.")
else:
    print("Image loaded successfully.")

# 2. Convert to Grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Converted to grayscale.")

# 3. Apply Thresholding to get a binary image (black and white)
_, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
print("Applied thresholding.")

# 4. Remove Noise using GaussianBlur
blurred_img = cv2.GaussianBlur(binary_img, (5, 5), 0)
print("Applied GaussianBlur.")

# 5. Edge Detection using Canny
edges = cv2.Canny(blurred_img, 100, 200)
print("Applied Canny edge detection.")

# 6. Find Contours (which should correspond to the ECG waveform)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("Error: No contours found.")
else:
    print(f"Found {len(contours)} contours.")

# 7. Extract the ECG Signal
# For simplicity, we assume the ECG signal is along the x-axis of the image.
# You might need to adjust the method to get the exact ECG signal depending on the quality of the image.

# Create an empty list to store the signal (amplitude)
amplitude_values = []
time_values = []

# Iterate over the contours and extract the signal
for contour in contours:
    for point in contour:
        x, y = point[0]
        amplitude_values.append(y)  # Y coordinate is the amplitude
        time_values.append(x)  # X coordinate is the time (pixel position)

print("Extracted ECG signal values.")

# 8. Convert the extracted values to a proper time scale (assuming 1 pixel = 1 ms for simplicity)
# You may need to adjust this scale based on the image's resolution and sampling rate.
time_values = np.array(time_values)
amplitude_values = np.array(amplitude_values)

# Rescale time axis if necessary
sampling_rate = 500  # Example: 500 Hz (samples per second)
time_values_rescaled = time_values / sampling_rate  # Converting pixel index to time

print("Rescaled time values.")

# Function to identify and split the ECG signal into P, QRS, T waves
def identify_ecg_waves(time_values, amplitude_values):
    # Find R peaks (highest peaks)
    r_peaks, _ = find_peaks(amplitude_values, height=np.max(amplitude_values) * 0.8)
    
    # Find Q and S valleys (lowest points before and after R peaks)
    q_peaks, _ = find_peaks(-amplitude_values, distance=50)
    s_peaks, _ = find_peaks(-amplitude_values, distance=50)
    
    # Find P waves (peaks before Q peaks)
    p_peaks, _ = find_peaks(amplitude_values, distance=50)
    
    return p_peaks, q_peaks, r_peaks, s_peaks

# Identify P, Q, R, S waves
p_peaks, q_peaks, r_peaks, s_peaks = identify_ecg_waves(time_values_rescaled, amplitude_values)

# Plot the Original ECG Signal with P, Q, R, S waves
plt.figure()
plt.plot(time_values_rescaled, amplitude_values, label='ECG Signal')
plt.plot(time_values_rescaled[p_peaks], amplitude_values[p_peaks], 'go', label='P Peaks')
plt.plot(time_values_rescaled[q_peaks], amplitude_values[q_peaks], 'ro', label='Q Peaks')
plt.plot(time_values_rescaled[r_peaks], amplitude_values[r_peaks], 'bo', label='R Peaks')
plt.plot(time_values_rescaled[s_peaks], amplitude_values[s_peaks], 'mo', label='S Peaks')
plt.title('ECG Signal with P, Q, R, S Waves')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Create a DataFrame to store the combined data
df_combined = pd.DataFrame({
    'Time': time_values_rescaled,
    'Amplitude': amplitude_values,
    'P_Peaks_Time': np.nan,
    'P_Peaks_Amplitude': np.nan,
    'Q_Peaks_Time': np.nan,
    'Q_Peaks_Amplitude': np.nan,
    'R_Peaks_Time': np.nan,
    'R_Peaks_Amplitude': np.nan,
    'S_Peaks_Time': np.nan,
    'S_Peaks_Amplitude': np.nan
})

# Fill in the P, Q, R, S peaks data
df_combined.loc[p_peaks, 'P_Peaks_Time'] = time_values_rescaled[p_peaks]
df_combined.loc[p_peaks, 'P_Peaks_Amplitude'] = amplitude_values[p_peaks]
df_combined.loc[q_peaks, 'Q_Peaks_Time'] = time_values_rescaled[q_peaks]
df_combined.loc[q_peaks, 'Q_Peaks_Amplitude'] = amplitude_values[q_peaks]
df_combined.loc[r_peaks, 'R_Peaks_Time'] = time_values_rescaled[r_peaks]
df_combined.loc[r_peaks, 'R_Peaks_Amplitude'] = amplitude_values[r_peaks]
df_combined.loc[s_peaks, 'S_Peaks_Time'] = time_values_rescaled[s_peaks]
df_combined.loc[s_peaks, 'S_Peaks_Amplitude'] = amplitude_values[s_peaks]

# Save the combined data to a CSV file
df_combined.to_csv(r'D:\Users\A.S.MUTHUVEL\Downloads\DAY 2\task2\ecg_combined_data.csv', index=False)
# Print confirmation message
print("ECG Signal Data saved to CSV (Combined Data).")
