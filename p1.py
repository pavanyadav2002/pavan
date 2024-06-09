import cv2

def capture_image(filename='vvv.jpg'):
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Display the resulting frame
        cv2.imshow('Capture Image', frame)
        
        # Wait for the user to press 'c' to capture the image or 'q' to exit
        key = cv2.waitKey(1)
        if key & 0xFF == ord('p'):
            # Save the captured image
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            break
        elif key & 0xFF == ord('q'):
            break
    
    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()


capture_image()