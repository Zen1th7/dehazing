import cv2
import numpy as np

def guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    mean_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))

    q = mean_a * I + mean_b
    return q

def dehaze_frame(frame, r=15, eps=0.001, omega=0.95, adjust_atm_light=True):
    # Convert the frame to float32 format
    frame = frame.astype('float32') / 255.0
    
    # Minimum value of intensity in each channel
    min_intensity = frame.min(axis=2)

    # Calculate the dark channel
    dark_channel = cv2.min(cv2.min(frame[:,:,0], frame[:,:,1]), frame[:,:,2])

    # Estimate atmospheric light
    atm_light = dark_channel.mean()
    
    # Optionally adjust atmospheric light
    if adjust_atm_light:
        atm_light *= 1.2  # You can adjust this factor as needed

    # Transmission map estimation
    min_filter = cv2.min(cv2.min(cv2.boxFilter(frame[:,:,0], -1, (r, r)), 
                                  cv2.boxFilter(frame[:,:,1], -1, (r, r))), 
                                  cv2.boxFilter(frame[:,:,2], -1, (r, r)))
    transmission_map = 1 - omega * min_filter / atm_light

    # Clamp transmission map to remove artifacts
    transmission_map = cv2.max(transmission_map, 0.1)

    # Refine the transmission map using guided filter
    guided_transmission_map = guided_filter(frame[:,:,2], transmission_map, r, eps)

    # Recovering scene radiance
    recovered_scene_radiance = np.zeros(frame.shape)
    for i in range(3):
        recovered_scene_radiance[:,:,i] = (frame[:,:,i] - atm_light) / guided_transmission_map + atm_light

    # Clamp the values to [0, 1]
    recovered_scene_radiance = np.clip(recovered_scene_radiance, 0, 1)

    # Convert back to uint8 format
    recovered_scene_radiance = (recovered_scene_radiance * 255).astype('uint8')

    return recovered_scene_radiance

# Read the video
cap = cv2.VideoCapture('how.mp4')

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video.")
    exit()

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the width and height of the video frames
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('dehazed_video.avi', fourcc, fps, (width, height))

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Dehaze the frame
    dehazed_frame = dehaze_frame(frame)

    # Write the dehazed frame to the output video
    out.write(dehazed_frame)

    # Display the dehazed frame
    cv2.imshow('Dehazed Frame', dehazed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()