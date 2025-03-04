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

# Function to apply FFT and save to CSV
def save_wave_fft_to_csv(time_values, amplitude_values, wave_name):
    fft_values = np.fft.fft(amplitude_values)
    frequencies = np.fft.fftfreq(len(amplitude_values), d=1/sampling_rate)
    
    df_wave = pd.DataFrame({
        'Frequency': frequencies,
        'Amplitude': np.abs(fft_values)
    })
    df_wave.to_csv(f'ecg_{wave_name}_frequency.csv', index=False)
    print(f"Saved {wave_name} wave frequency data to CSV.")

# Save FFT data for each wave
save_wave_fft_to_csv(time_values_rescaled[p_peaks], amplitude_values[p_peaks], 'P')
save_wave_fft_to_csv(time_values_rescaled[q_peaks], amplitude_values[q_peaks], 'Q')
save_wave_fft_to_csv(time_values_rescaled[r_peaks], amplitude_values[r_peaks], 'R')
save_wave_fft_to_csv(time_values_rescaled[s_peaks], amplitude_values[s_peaks], 'S')

# 10. Apply FFT to the Signal to Convert to Frequency Domain
fft_values = np.fft.fft(amplitude_values)
frequencies = np.fft.fftfreq(len(amplitude_values), d=1/sampling_rate)

print("Applied FFT to the signal.")

# 11. Plot the Frequency Domain (FFT)
plt.figure()
plt.plot(frequencies, np.abs(fft_values))
plt.title('ECG Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 10)  # Limit to 10 Hz (for heart-related frequencies)
plt.show()

# 12. Save Time Domain Data to CSV
time_data = time_values_rescaled  # Time values
amplitude_data = amplitude_values  # Amplitude values

df_time = pd.DataFrame({'Time': time_data, 'Amplitude': amplitude_data})
df_time.to_csv('ecg_signal_time.csv', index=False)

# 13. Save Frequency Domain Data to CSV
freq_data = frequencies  # Frequency values from FFT
amplitude_freq_data = np.abs(fft_values)  # Magnitude of the frequency components

df_freq = pd.DataFrame({'Frequency': freq_data, 'Amplitude': amplitude_freq_data})
df_freq.to_csv('ecg_signal_frequency.csv', index=False)

# Save P, Q, R, S wave data to CSV
df_waves = pd.DataFrame({
    'P_Peaks_Time': time_values_rescaled[p_peaks],
    'P_Peaks_Amplitude': amplitude_values[p_peaks],
    'Q_Peaks_Time': time_values_rescaled[q_peaks],
    'Q_Peaks_Amplitude': amplitude_values[q_peaks],
    'R_Peaks_Time': time_values_rescaled[r_peaks],
    'R_Peaks_Amplitude': amplitude_values[r_peaks],
    'S_Peaks_Time': time_values_rescaled[s_peaks],
    'S_Peaks_Amplitude': amplitude_values[s_peaks]
})
df_waves.to_csv('ecg_signal_waves.csv', index=False)

# Print confirmation message
print("ECG Signal Data saved to CSV (Time Domain, Frequency Domain, and Waves).")