import cv2

# Create a VideoCapture object
camera = cv2.VideoCapture(2)

# Initialize a counter
count = 0

while count < 50:
    # Capture a frame
    ret, frame = camera.read()

    # Save the frame as a JPG image
    cv2.imwrite(f"fotos50/tomate/frame_{count}.jpg", frame)

    # Increase the counter
    count += 1

# Release the camera
camera.release()
# Close all windows
cv2.destroyAllWindows()