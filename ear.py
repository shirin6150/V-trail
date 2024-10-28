import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, min_detection_confidence=0.7
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load earring PNG with alpha channel (ensure it has transparency)
earring_img = cv2.imread('Earrings-1.png', cv2.IMREAD_UNCHANGED)

# Check if the image was loaded successfully
if earring_img is None:
    raise FileNotFoundError("The earring image could not be loaded. Please check the file path.")

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return int(np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2))

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """
    Overlays a transparent PNG image on a background at (x, y) with optional resizing.
    """
    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size)

    h, w, _ = overlay.shape
    bg_h, bg_w, _ = background.shape

    # Ensure the overlay fits within the background dimensions
    if y + h > bg_h or x + w > bg_w or x < 0 or y < 0:
        h = min(h, bg_h - y)
        w = min(w, bg_w - x)
        overlay = overlay[:h, :w]

    roi = background[y:y+h, x:x+w]

    # Extract overlay RGB and alpha channels
    overlay_rgb = overlay[..., :3]  # RGB
    if overlay.shape[2] == 4:
        mask = overlay[..., 3:] / 255.0  # Alpha
    else:
        raise ValueError("Overlay image does not have an alpha channel.")

    # Ensure mask and roi have compatible shapes
    if mask.shape[:2] != roi.shape[:2]:
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))

    # Ensure mask has the correct number of channels
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]

    # Blend the overlay with the background
    background[y:y+h, x:x+w] = (1.0 - mask) * roi + mask * overlay_rgb
    return background

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame and convert to RGB for MediaPipe processing
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with FaceMesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract key landmarks: Left and Right earlobe, and ear tops
            h, w, _ = frame.shape

            # Left ear: landmark 234 (ear top) and 177 (earlobe)
            left_ear_top = (int(face_landmarks.landmark[234].x * w), 
                            int(face_landmarks.landmark[234].y * h))
            left_earlobe = (int(face_landmarks.landmark[177].x * w), 
                            int(face_landmarks.landmark[177].y * h))

            # Right ear: landmark 454 (ear top) and 401 (earlobe)
            right_ear_top = (int(face_landmarks.landmark[454].x * w), 
                             int(face_landmarks.landmark[454].y * h))
            right_earlobe = (int(face_landmarks.landmark[401].x * w), 
                             int(face_landmarks.landmark[401].y * h))

            # Calculate earring size based on distance between ear top and earlobe
            left_earring_size = calculate_distance(left_ear_top, left_earlobe)
            right_earring_size = calculate_distance(right_ear_top, right_earlobe)

            # Adjust the earring position slightly for more accuracy
            left_x = left_earlobe[0] - left_earring_size // 2
            left_y = left_earlobe[1] - left_earring_size // 2

            right_x = right_earlobe[0] - right_earring_size // 2
            right_y = right_earlobe[1] - right_earring_size // 2

            # Overlay the earrings on both ears
            frame = overlay_transparent(
                frame, earring_img, left_x, left_y, (left_earring_size, left_earring_size)
            )
            frame = overlay_transparent(
                frame, earring_img, right_x, right_y, (right_earring_size, right_earring_size)
            )

    # Display the output frame
    cv2.imshow('Virtual Earring Try-On', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()




import cv2
img = cv2.imread('ear1.png')  # Use any sample image you have
cv2.imshow('Test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()




# import cv2

# path = r"C:\Users\shiri\OneDrive\Documents\VS Code\virtual trail\ear.png"
# img = cv2.imread(path)

# if img is None:
#     print("Failed to load image.")
# else:
#     print("Image loaded successfully!")
