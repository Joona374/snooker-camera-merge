import cv2
import numpy as np

class SnookerTableMerger:
    def __init__(self):
        self.points_left = []
        self.points_right = []
        self.current_points = []
        self.current_image = None
        self.adjustment_mode = False
        self.selected_camera = 0  # 0 = left, 1 = right
        self.selected_point = 0
        self.step_size = 1  # pixels to move per keypress
        
        self.left_labels = [
            "Top-Left Corner",
            "Bottom-Left Corner",
            "Top Middle Pocket", 
            "Bottom Middle Pocket"
        ]
        
        self.right_labels = [
            "Top-Right Corner",
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
            cv2.putText(self.current_image, f"{label_idx+1}", 
                       (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Click Points", self.current_image)
            if label_idx < 4:
                print(f"Point {len(self.current_points)} clicked at ({x}, {y})")

    def collect_points(self, image, camera_name):
        self.current_points = []
        self.current_image = image.copy()
        
        # Choose the right labels based on camera
        labels = self.left_labels if "LEFT" in camera_name else self.right_labels
        
        print(f"\n=== {camera_name} ===")
        print("Click the following points in order:")
        for i, label in enumerate(labels):
            print(f"{i+1}. {label}")
        print("Press SPACE when done, ESC to cancel")
        
        cv2.imshow("Click Points", self.current_image)
        cv2.setMouseCallback("Click Points", self.mouse_callback)
        
        while len(self.current_points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
            elif key == 32 and len(self.current_points) == 4:  # SPACE
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

    def calculate_output_dimensions(self, pts1, pts2, img1_shape, img2_shape):
        """Calculate proper output dimensions based on table coverage in each camera"""
        
        # For left camera: distance from left corner to middle pocket
        left_width = abs(pts1[2][0] - pts1[0][0])  # Top middle pocket X - Top left corner X
        
        # For right camera: distance from middle pocket to right corner  
        right_width = abs(pts2[0][0] - pts2[2][0])  # Top right corner X - Top middle pocket X
        
        # Height should be consistent - use the larger of the two
        left_height = abs(pts1[1][1] - pts1[0][1])  # Bottom left - Top left
        right_height = abs(pts2[1][1] - pts2[0][1])  # Bottom right - Top right
        
        output_height = max(left_height, right_height)
        output_width = left_width + right_width
        
        return int(output_width), int(output_height), int(left_width), int(right_width)

    def create_merged_image(self, img1, img2, pts1, pts2):
        """Create the merged snooker table image with proper aspect ratio"""
        
        # Calculate proper dimensions
        output_width, output_height, left_width, right_width = self.calculate_output_dimensions(
            pts1, pts2, img1.shape, img2.shape)
        
        print(f"Calculated output dimensions: {output_width}x{output_height}")
        print(f"Left section: {left_width}px, Right section: {right_width}px")
        
        # Left camera maps to left portion
        left_target = np.array([
            [0, 0],                    # Top-Left Corner → Top-Left of output
            [0, output_height],        # Bottom-Left Corner → Bottom-Left of output  
            [left_width, 0],           # Top Middle Pocket → Top at left_width
            [left_width, output_height]  # Bottom Middle Pocket → Bottom at left_width
        ], dtype=np.float32)
        
        # Right camera maps to right portion
        right_target = np.array([
            [output_width, 0],         # Top-Right Corner → Top-Right of output
            [output_width, output_height], # Bottom-Right Corner → Bottom-Right of output
            [left_width, 0],           # Top Middle Pocket → Top at left_width (same as left camera)
            [left_width, output_height]  # Bottom Middle Pocket → Bottom at left_width
        ], dtype=np.float32)
        
        # Create perspective transforms
        H1 = cv2.getPerspectiveTransform(pts1, left_target)
        H2 = cv2.getPerspectiveTransform(pts2, right_target)
        
        # Warp both images
        warped1 = cv2.warpPerspective(img1, H1, (output_width, output_height))
        warped2 = cv2.warpPerspective(img2, H2, (output_width, output_height))
        
        # Create masks for left and right portions
        mask_left = np.zeros((output_height, output_width), dtype=np.uint8)
        mask_right = np.zeros((output_height, output_width), dtype=np.uint8)
        
        mask_left[:, :left_width] = 255
        mask_right[:, left_width:] = 255
        
        # Apply masks
        left_masked = cv2.bitwise_and(warped1, warped1, mask=mask_left)
        right_masked = cv2.bitwise_and(warped2, warped2, mask=mask_right)
        
        # Combine the images
        result = cv2.add(left_masked, right_masked)
        
        return result, H1, H2, output_width, output_height, left_width

    def draw_adjustment_overlay(self, image, pts1, pts2):
        """Draw adjustment overlay showing current points and selection"""
        overlay = image.copy()
        
        # Draw current camera points
        current_pts = pts1 if self.selected_camera == 0 else pts2
        current_labels = self.left_labels if self.selected_camera == 0 else self.right_labels
        
        for i, (point, label) in enumerate(zip(current_pts, current_labels)):
            x, y = int(point[0]), int(point[1])
            
            # Highlight selected point
            if i == self.selected_point:
                cv2.circle(overlay, (x, y), 10, (0, 255, 255), 3)  # Yellow circle for selected
                cv2.putText(overlay, f"SELECTED: {label}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.circle(overlay, (x, y), 7, (0, 255, 0), 2)  # Green for others
            
            # Point number
            cv2.putText(overlay, str(i+1), (x+15, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        camera_name = "LEFT" if self.selected_camera == 0 else "RIGHT"
        instructions = [
            f"Camera: {camera_name} | Point: {self.selected_point+1}/{len(current_pts)}",
            "Arrow Keys: Move point | Tab: Next point | Space: Switch camera",
            f"1-5: Step size ({self.step_size}px) | Enter: Done | ESC: Cancel"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(overlay, instruction, (10, image.shape[0] - 60 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay

    def live_adjustment_mode(self, cam1, cam2, pts1, pts2):
        """Live adjustment mode for fine-tuning points"""
        print("\n" + "="*60)
        print("LIVE ADJUSTMENT MODE")
        print("="*60)
        print("Controls:")
        print("  Arrow Keys: Move selected point")
        print("  Tab: Select next point")
        print("  Space: Switch between cameras")
        print("  1-5: Change step size (1, 2, 5, 10, 20 pixels)")
        print("  Enter: Apply changes and continue")
        print("  ESC: Cancel changes")
        print("="*60)
        
        # Create working copies
        adj_pts1 = pts1.copy()
        adj_pts2 = pts2.copy()
        
        self.selected_camera = 0
        self.selected_point = 0
        self.step_size = 1
        
        while True:
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()
            
            if not ret1 or not ret2:
                print("Error reading from cameras.")
                return None, None
            
            # Create merged image with current points
            try:
                merged, _, _, _, _, _ = self.create_merged_image(frame1, frame2, adj_pts1, adj_pts2)
                
                # Draw overlay on the frame being adjusted
                if self.selected_camera == 0:
                    frame_with_overlay = self.draw_adjustment_overlay(frame1, adj_pts1, adj_pts2)
                    cv2.imshow("Camera View (Adjusting)", frame_with_overlay)
                else:
                    frame_with_overlay = self.draw_adjustment_overlay(frame2, adj_pts1, adj_pts2)
                    cv2.imshow("Camera View (Adjusting)", frame_with_overlay)
                
                cv2.imshow("Live Merge Preview", merged)
                
            except Exception as e:
                print(f"Error creating merge preview: {e}")
                continue
            
            key = cv2.waitKey(1) & 0xFF
            
            # Get current points array
            current_pts = adj_pts1 if self.selected_camera == 0 else adj_pts2
            
            # Arrow key controls
            if key == 81 or key == 0:  # Left arrow (may vary by system)
                current_pts[self.selected_point][0] -= self.step_size
            elif key == 83 or key == 1:  # Right arrow  
                current_pts[self.selected_point][0] += self.step_size
            elif key == 82 or key == 2:  # Up arrow
                current_pts[self.selected_point][1] -= self.step_size
            elif key == 84 or key == 3:  # Down arrow
                current_pts[self.selected_point][1] += self.step_size
            
            # Alternative WASD controls
            elif key == ord('w'):  # Up
                current_pts[self.selected_point][1] -= self.step_size
            elif key == ord('s'):  # Down
                current_pts[self.selected_point][1] += self.step_size
            elif key == ord('a'):  # Left
                current_pts[self.selected_point][0] -= self.step_size
            elif key == ord('d'):  # Right
                current_pts[self.selected_point][0] += self.step_size
            
            # Tab - next point
            elif key == 9:  # Tab
                self.selected_point = (self.selected_point + 1) % len(current_pts)
            
            # Space - switch camera
            elif key == 32:  # Space
                self.selected_camera = 1 - self.selected_camera
                self.selected_point = 0  # Reset to first point
            
            # Number keys for step size
            elif key == ord('1'):
                self.step_size = 1
            elif key == ord('2'):
                self.step_size = 2
            elif key == ord('3'):
                self.step_size = 5
            elif key == ord('4'):
                self.step_size = 10
            elif key == ord('5'):
                self.step_size = 20
            
            # Enter - apply changes
            elif key == 13:  # Enter
                cv2.destroyWindow("Camera View (Adjusting)")
                cv2.destroyWindow("Live Merge Preview")
                return adj_pts1, adj_pts2
            
            # ESC - cancel
            elif key == 27:  # ESC
                cv2.destroyWindow("Camera View (Adjusting)")
                cv2.destroyWindow("Live Merge Preview")
                return pts1, pts2  # Return original points

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
    test_merge, H1, H2, output_width, output_height, left_width = merger.create_merged_image(frame1, frame2, pts1, pts2)
    
    cv2.imshow("Test Merge", test_merge)
    print("Test merge created.")
    print("Press 'a' for adjustment mode, any other key for live feed, or ESC to exit.")
    
    key = cv2.waitKey(0)
    cv2.destroyWindow("Test Merge")
    
    if key == 27:  # ESC
        cv2.destroyAllWindows()
        cam1.release()
        cam2.release()
        return
    elif key == ord('a'):  # Adjustment mode
        adjusted_pts1, adjusted_pts2 = merger.live_adjustment_mode(cam1, cam2, pts1, pts2)
        if adjusted_pts1 is not None:
            pts1, pts2 = adjusted_pts1, adjusted_pts2
            # Recreate transforms with adjusted points
            test_merge, H1, H2, output_width, output_height, left_width = merger.create_merged_image(frame1, frame2, pts1, pts2)
            print("Points adjusted! Starting live feed...")
        else:
            print("Adjustment cancelled, using original points.")
    
    # Live merging
    print("\n" + "="*50)
    print("LIVE MERGE - Press 'q' to quit, 'a' to adjust again")
    print("="*50)
    
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
        
        mask_left[:, :left_width] = 255
        mask_right[:, left_width:] = 255
        
        left_masked = cv2.bitwise_and(warped1, warped1, mask=mask_left)
        right_masked = cv2.bitwise_and(warped2, warped2, mask=mask_right)
        
        merged = cv2.add(left_masked, right_masked)
        
        # Draw center line for reference
        cv2.line(merged, (left_width, 0), (left_width, output_height), (0, 255, 255), 2)
        
        cv2.imshow("Snooker Table Merge", merged)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):  # Re-enter adjustment mode
            print("\nRe-entering adjustment mode...")
            cv2.destroyWindow("Snooker Table Merge")
            adjusted_pts1, adjusted_pts2 = merger.live_adjustment_mode(cam1, cam2, pts1, pts2)
            if adjusted_pts1 is not None:
                pts1, pts2 = adjusted_pts1, adjusted_pts2
                # Recreate transforms
                ret1, frame1 = cam1.read()
                ret2, frame2 = cam2.read()
                if ret1 and ret2:
                    _, H1, H2, output_width, output_height, left_width = merger.create_merged_image(frame1, frame2, pts1, pts2)
                    print("Points re-adjusted! Continuing live feed...")
            print("Press 'q' to quit, 'a' to adjust again")
    
    # Adjustment mode
    adj_pts1, adj_pts2 = merger.live_adjustment_mode(cam1, cam2, pts1, pts2)
    
    # Final merge with adjusted points
    print("\nCreating final merge with adjusted points...")
    final_merge, _, _, _, _, _ = merger.create_merged_image(frame1, frame2, adj_pts1, adj_pts2)
    
    cv2.imshow("Final Merge", final_merge)
    cv2.waitKey(0)
    
    # Cleanup
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
