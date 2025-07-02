import cv2
import numpy as np

class SnookerTableMerger:
    def __init__(self):
        self.points_left = []
        self.points_right = []
        self.current_points = []
        self.current_image = None
        self.point_labels = [
            "Top-Left Corner",
            "Top-Right Corner", 
            "Bottom-Left Corner",
            "Bottom-Right Corner",
            "Top Middle Pocket",
            "Bottom Middle Pocket"
        ]
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            cv2.circle(self.current_image, (x, y), 5, (0, 255, 0), -1)
            
            # Add label text
            label_idx = len(self.current_points) - 1
            if label_idx < len(self.point_labels):
                cv2.putText(self.current_image, f"{label_idx+1}", 
                           (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Click Points", self.current_image)
            print(f"Point {len(self.current_points)}: {self.point_labels[label_idx] if label_idx < len(self.point_labels) else 'Unknown'}")

    def collect_points(self, image, camera_name):
        self.current_points = []
        self.current_image = image.copy()
        
        print(f"\n=== {camera_name} ===")
        print("Click the following points in order:")
        for i, label in enumerate(self.point_labels):
            print(f"{i+1}. {label}")
        print("Press SPACE when done, ESC to cancel")
        
        cv2.imshow("Click Points", self.current_image)
        cv2.setMouseCallback("Click Points", self.mouse_callback)
        
        while len(self.current_points) < 6:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
            elif key == 32 and len(self.current_points) == 6:  # SPACE
                break
                
        cv2.destroyWindow("Click Points")
        return np.array(self.current_points, dtype=np.float32)

    def create_standard_rectangle(self, width=800, height=400):
        """Create a standard rectangle for the snooker table output"""
        return np.array([
            [0, 0],           # Top-left
            [width, 0],       # Top-right  
            [0, height],      # Bottom-left
            [width, height],  # Bottom-right
            [width//2, 0],    # Top middle
            [width//2, height] # Bottom middle
        ], dtype=np.float32)

    def create_perspective_transform(self, src_points, dst_points):
        """Create perspective transform using the 4 corner points"""
        src_corners = src_points[:4]  # First 4 points are corners
        dst_corners = dst_points[:4]
        return cv2.getPerspectiveTransform(src_corners, dst_corners)

    def create_merged_image(self, img1, img2, pts1, pts2, output_width=1600, output_height=400):
        """Create the merged snooker table image"""
        
        # Create target points for left half (0 to middle)
        left_target = np.array([
            [0, 0],                    # Top-left
            [output_width//2, 0],      # Top-middle 
            [0, output_height],        # Bottom-left
            [output_width//2, output_height], # Bottom-middle
            [output_width//2, 0],      # Top middle pocket
            [output_width//2, output_height]  # Bottom middle pocket
        ], dtype=np.float32)
        
        # Create target points for right half (middle to end)
        right_target = np.array([
            [output_width//2, 0],      # Top-middle (maps to top-left of right camera)
            [output_width, 0],         # Top-right
            [output_width//2, output_height], # Bottom-middle (maps to bottom-left of right camera)  
            [output_width, output_height],    # Bottom-right
            [output_width//2, 0],      # Top middle pocket
            [output_width//2, output_height]  # Bottom middle pocket
        ], dtype=np.float32)
        
        # Create perspective transforms
        H1 = cv2.getPerspectiveTransform(pts1[:4], left_target[:4])
        H2 = cv2.getPerspectiveTransform(pts2[:4], right_target[:4])
        
        # Warp both images
        warped1 = cv2.warpPerspective(img1, H1, (output_width, output_height))
        warped2 = cv2.warpPerspective(img2, H2, (output_width, output_height))
        
        # Create masks for left and right halves
        mask_left = np.zeros((output_height, output_width), dtype=np.uint8)
        mask_right = np.zeros((output_height, output_width), dtype=np.uint8)
        
        mask_left[:, :output_width//2] = 255
        mask_right[:, output_width//2:] = 255
        
        # Apply masks
        left_masked = cv2.bitwise_and(warped1, warped1, mask=mask_left)
        right_masked = cv2.bitwise_and(warped2, warped2, mask=mask_right)
        
        # Combine the images
        result = cv2.add(left_masked, right_masked)
        
        return result, H1, H2

def main():
    merger = SnookerTableMerger()
    
    # Open cameras
    print("Opening cameras...")
    cam1 = cv2.VideoCapture(0)  # Left camera
    cam2 = cv2.VideoCapture(2)  # Right camera
    
    if not cam1.isOpened() or not cam2.isOpened():
        print("Error: Could not open one or both cameras.")
        return
    
    # Capture calibration frames
    print("Capturing calibration frames...")
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()
    
    if not ret1 or not ret2:
        print("Error: Could not read from cameras.")
        return
    
    print("Frame sizes:")
    print(f"Camera 1: {frame1.shape}")
    print(f"Camera 2: {frame2.shape}")
    
    # Collect points for both cameras
    print("\n" + "="*50)
    print("CALIBRATION PHASE")
    print("="*50)
    
    pts1 = merger.collect_points(frame1, "LEFT CAMERA")
    if pts1 is None:
        print("Calibration cancelled.")
        return
        
    pts2 = merger.collect_points(frame2, "RIGHT CAMERA") 
    if pts2 is None:
        print("Calibration cancelled.")
        return
    
    print(f"\nLeft camera points: {pts1}")
    print(f"Right camera points: {pts2}")
    
    # Create initial merged image to test
    print("\nCreating test merge...")
    test_merge, H1, H2 = merger.create_merged_image(frame1, frame2, pts1, pts2)
    
    cv2.imshow("Test Merge", test_merge)
    print("Test merge created. Press any key to start live feed, or ESC to exit.")
    
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        cv2.destroyAllWindows()
        cam1.release()
        cam2.release()
        return
    
    cv2.destroyWindow("Test Merge")
    
    # Live merging
    print("\n" + "="*50)
    print("LIVE MERGE - Press 'q' to quit")
    print("="*50)
    
    output_width = 1600
    output_height = 400
    
    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        
        if not ret1 or not ret2:
            print("Error reading from cameras.")
            break
        
        # Apply the same transforms
        warped1 = cv2.warpPerspective(frame1, H1, (output_width, output_height))
        warped2 = cv2.warpPerspective(frame2, H2, (output_width, output_height))
        
        # Create masks and combine
        mask_left = np.zeros((output_height, output_width), dtype=np.uint8)
        mask_right = np.zeros((output_height, output_width), dtype=np.uint8)
        
        mask_left[:, :output_width//2] = 255
        mask_right[:, output_width//2:] = 255
        
        left_masked = cv2.bitwise_and(warped1, warped1, mask=mask_left)
        right_masked = cv2.bitwise_and(warped2, warped2, mask=mask_right)
        
        merged = cv2.add(left_masked, right_masked)
        
        # Draw center line for reference
        cv2.line(merged, (output_width//2, 0), (output_width//2, output_height), (0, 255, 255), 2)
        
        cv2.imshow("Snooker Table Merge", merged)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
