import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load bracelet PNG with alpha channel (ensure it has transparency)
bracelet_img = cv2.imread(r"C:\Users\shiri\OneDrive\Documents\VS Code\virtual trail\assets\bracelet_only.png", cv2.IMREAD_UNCHANGED)

# Check if the bracelet image was loaded successfully
if bracelet_img is None:
    raise FileNotFoundError("The bracelet image could not be loaded. Please check the file path.")

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return int(np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2))

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """Overlays a transparent PNG on a background at (x, y) with optional resizing."""
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

    # Ensure roi is valid before processing
    if roi.size == 0:
        return background  # No valid region to overlay

    # Extract overlay RGB and alpha channels
    overlay_rgb = overlay[..., :3]  # RGB
    mask = overlay[..., 3:] / 255.0  # Alpha

    # Resize mask if needed to match ROI size
    if mask.shape[:2] != roi.shape[:2]:
        try:
            mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
        except Exception as e:
            print(f"Error resizing mask: {e}")
            return background  # Return original background if resizing fails

    # Add dimension if mask is 2D
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]

    # Blend the overlay with the background using the mask
    blended = roi * (1 - mask) + overlay_rgb * mask
    background[y:y+h, x:x+w] = blended.astype(np.uint8)
    return background

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror view and convert to RGB for MediaPipe
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape

            # Get wrist and index finger base landmarks
            wrist = hand_landmarks.landmark[0]  # Wrist landmark
            index_finger_base = hand_landmarks.landmark[5]  # Index finger base

            # Convert landmarks to pixel coordinates
            wrist_point = (int(wrist.x * w), int(wrist.y * h))
            index_point = (int(index_finger_base.x * w), int(index_finger_base.y * h))

            # Calculate bracelet size based on distance between wrist and index finger base
            bracelet_size = calculate_distance(wrist_point, index_point)

            # Adjust the bracelet position
            x = wrist_point[0] - bracelet_size // 2
            y = wrist_point[1] - bracelet_size // 2

            # Overlay the bracelet on the wrist
            frame = overlay_transparent(frame, bracelet_img, x, y, (bracelet_size, bracelet_size))

    # Display the output frame
    cv2.imshow('Virtual Bracelet Try-On', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()



# from PIL import Image

# # Load your image
# image_path = 'bracelet2.png'  # Replace with your image path
# img = Image.open(image_path)

# # Convert image to RGBA (adds alpha channel)
# img = img.convert("RGBA")

# # Optionally, set a transparent background
# # Create a new image with a transparent background
# width, height = img.size
# transparent_background = Image.new("RGBA", (width, height), (0, 0, 0, 0))

# # Paste the original image onto the transparent background
# transparent_background.paste(img, (0, 0), img)

# # Save as PNG
# transparent_background.save('bracelet_with_alpha.png', 'PNG')


# File: isolate_bracelet.py

# from rembg import remove
# from PIL import Image
# import cv2
# import numpy as np
# import io

# def remove_background(input_path: str) -> Image.Image:
#     """Remove the background using rembg and return as a PIL Image."""
#     with open(input_path, "rb") as img_file:
#         input_image = img_file.read()

#     output_image = remove(input_image)  # Remove background with rembg

#     # Convert to PIL Image for further processing
#     pil_image = Image.open(io.BytesIO(output_image)).convert("RGBA")
#     return pil_image

# def extract_bracelet(pil_image: Image.Image, output_path: str):
#     """Detect and extract the bracelet using contour detection."""
#     # Convert PIL image to NumPy array
#     np_image = np.array(pil_image)

#     # Extract alpha channel (mask) from RGBA image
#     alpha_channel = np_image[:, :, 3]  # Get transparency mask

#     # Threshold the alpha channel to create a binary mask
#     _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

#     # Find contours in the binary mask
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Filter contours to find the bracelet (largest by area)
#     largest_contour = max(contours, key=cv2.contourArea)

#     # Create a mask for the bracelet
#     mask = np.zeros_like(binary_mask)
#     cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

#     # Apply the mask to the original image (RGBA)
#     bracelet_only = cv2.bitwise_and(np_image, np_image, mask=mask)

#     # Convert the result back to a PIL image and save it
#     result_pil = Image.fromarray(bracelet_only, mode="RGBA")
#     result_pil.save(output_path, "PNG")
#     print(f"Bracelet extracted and saved at: {output_path}")

# if __name__ == "__main__":
#     input_image_path = r"C:\Users\shiri\OneDrive\Documents\VS Code\virtual trail\assets\bracelet 3.png"  # Input image path
#     output_image_path = r"C:\Users\shiri\OneDrive\Documents\VS Code\virtual trail\assets\bracelet_nbg 3.png" # Output image path

#     # Step 1: Remove the background using rembg
#     pil_image = remove_background(input_image_path)

#     # Step 2: Extract only the bracelet (largest contour)
#     extract_bracelet(pil_image, output_image_path)


# import cv2
# import numpy as np

# def remove_skin_background(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Define HSV range for skin color
#     lower_skin = np.array([0, 20, 70], dtype=np.uint8)
#     upper_skin = np.array([20, 255, 255], dtype=np.uint8)

#     # Create a mask for skin color
#     skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

#     # Invert the skin mask to focus on non-skin regions (like the bracelet)
#     non_skin_mask = cv2.bitwise_not(skin_mask)

#     # Apply the mask to the original image
#     bracelet_only = cv2.bitwise_and(image, image, mask=non_skin_mask)

#     # Optional: Improve mask with morphology operations (reduce noise)
#     kernel = np.ones((5, 5), np.uint8)
#     bracelet_only = cv2.morphologyEx(bracelet_only, cv2.MORPH_CLOSE, kernel)

#     # Display the original and processed images
#     cv2.imshow("Original Image", image)
#     cv2.imshow("Bracelet Only", bracelet_only)

#     # Save the result
#     cv2.imwrite("bracelet_extracted.png", bracelet_only)

#     # Wait for a key press and close windows
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Run the function
# image_path = r"C:\Users\shiri\OneDrive\Documents\VS Code\virtual trail\assets\bracelet_nbg 3.png"
# remove_skin_background(image_path)
