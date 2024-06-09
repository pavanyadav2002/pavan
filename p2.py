import cv2
import pytesseract
import numpy as np

# Configure the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'/Users/pavanyadav/pytesseract'

# Function to adjust brightness and contrast
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    beta = brightness
    alpha = contrast / 127.0 + 1.0
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image

# Function to sharpen the image
def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen_image = cv2.filter2D(image, -1, kernel)
    return sharpen_image

def binarize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, binary_image = cv2.threshold(gray, 200, 230, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary_image

def denoise(image):
    kernal = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernal, iterations=1)
    kernal = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernal, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernal)
    image = cv2.medianBlur(image, 3)
    return image

def thin_font(image):
    image = cv2.bitwise_not(image)
    kernal = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernal, iterations=1)
    image = cv2.bitwise_not(image)
    return image
def enhance_text_image(image):
    new_image = adjust_brightness_contrast(image, brightness=30, contrast=30)
    blurred_image = cv2.GaussianBlur(new_image, (5, 5), 0)
    sharpen_image = sharpen(blurred_image)
    blurred_sharpen_image = cv2.GaussianBlur(sharpen_image, (5, 5), 0)
    binary_image = binarize(blurred_sharpen_image)
    image = denoise(binary_image)
    return image

def capture_and_process():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Couldn't access the webcam.")
        return

    ret, frame = cap.read()

    if not ret:
        print("Error: Couldn't capture a frame.")
        return

    cv2.imshow('Captured Image', frame)
    cv2.waitKey(0)

    enhanced_frame = enhance_text_image(frame)

    cv2.imshow('Enhanced Image', enhanced_frame)
    cv2.waitKey(0)

    cv2.imwrite('captured_image.jpg', frame)
    cv2.imwrite('enhanced_text_image.jpg', enhanced_frame)
    print("Image captured and enhanced successfully.")

    extracted_text = perform_ocr('enhanced_text_image.jpg')
    print("Extracted Text:")
    print(extracted_text)

    cap.release()
    cv2.destroyAllWindows()

def perform_ocr(image_path):
    image = cv2.imread(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text

capture_and_process()