# Import the OpenCV library
import cv2

# Load Haar Cascades for face and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Initialize the camera capture
cap = cv2.VideoCapture(0)

# Start an infinite loop to capture video frames
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Draw a red rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Get the region of interest (ROI) in grayscale and color
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect smiles in the ROI
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
        smile_detected = False

        # Calculate face detection confidence
        face_area = w * h

        # Loop through the detected smiles and calculate smile detection confidence
        for (sx, sy, sw, sh) in smiles:
            smile_area = sw * sh
            confidence = (smile_area / face_area) * 100

            # If smile area is a significant portion of the face area, consider it a smile
            if confidence > 10:
                smile_detected = True
                # Draw a green rectangle around the smile
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

        # Display face detection confidence
        face_confidence = f"Face Detected"
        cv2.putText(frame, face_confidence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display smile detection confidence
        if smile_detected:
            smile_confidence = f"Smile Detected: {100-int(confidence)}%"
        else:
            smile_confidence = "No Smile"
        cv2.putText(frame, smile_confidence, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with confidence scores
    cv2.imshow('Smile Detection', frame)

    # Check for the spacebar key press to exit the loop
    key = cv2.waitKey(1)
    if key == 32:  # Spacebar key
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
