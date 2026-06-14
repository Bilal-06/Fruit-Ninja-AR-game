import cv2
import mediapipe as mp

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)



    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with mediapipe hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for point in landmarks.landmark:
                # Extract hand landmarks (x, y, z)
                x, y, _ = int(point.x * frame.shape[1]), int(point.y * frame.shape[0]), int(point.z * frame.shape[0])

                # Draw a circle at each hand landmark
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('u'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
