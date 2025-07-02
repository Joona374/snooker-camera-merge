import cv2
import numpy as np

# Mouse callback to collect points
def click_points(event, x, y, flags, params):
    image, points = params
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", image)

def collect_points(img, prompt):
    points = []
    temp_img = img.copy()
    cv2.imshow("Image", temp_img)
    cv2.setMouseCallback("Image", click_points, (temp_img, points))
    print(f"Please click the 4 points for: {prompt}")
    while len(points) < 4:
        cv2.waitKey(1)
    cv2.destroyWindow("Image")
    return np.array(points, dtype=np.float32)

def main():
    # Open cameras
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(2)

    # Capture one frame from each camera for calibration
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        print("Error: Could not read from one or both cameras.")
        return

    # Collect points interactively on the captured frames
    pts1 = collect_points(frame1, "Left Camera (Top-Left, Top-Middle, Bottom-Left, Bottom-Middle)")
    pts2 = collect_points(frame2, "Right Camera (Top-Left, Top-Middle, Bottom-Left, Bottom-Middle)")

    # Compute perspective transform to align right camera to left camera view
    H = cv2.getPerspectiveTransform(pts2, pts1)

    # Prepare for live streaming and stitching
    width = frame1.shape[1]
    height = frame1.shape[0]

    print("Starting live stitched feed. Press 'q' to quit.")

    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        if not ret1 or not ret2:
            print("Error reading from cameras.")
            break

        # Warp right camera frame
        warped = cv2.warpPerspective(frame2, H, (width, height))

        # Calculate seam line between mid-pocket points on left camera (pts1)
        x1, y1 = pts1[1]
        x3, y3 = pts1[3]
        rows = height
        seam_x = np.array([int(x1 + (y - y1) * (x3 - x1) / (y3 - y1)) for y in range(rows)])

        # Create composite image along seam
        composite = np.zeros_like(frame1)
        for y in range(rows):
            sx = seam_x[y]
            composite[y, :sx] = frame1[y, :sx]
            composite[y, sx:] = warped[y, sx:]

        cv2.imshow("Stitched Live Feed", composite)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
