import torch
import numpy as np
import time
from enum import Enum
import cv2

# Available Models
class ModelType(Enum):
    DPT_LARGE = "DPT_Large" # Slow but powerful
    DPT_HYBRID = "DPT_Hybrid" # Balance between speed and power
    MIDAS_SMALL = "MiDaS_small" # Fast but not powerful

class MidasDetector:
    # Constructor
    def __init__(self, model_type=ModelType.MIDAS_SMALL): # Use small model for faster speed
        print("Loading MiDaS model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("isl-org/MiDaS", model_type.value).to(self.device).eval()

        # Transform Input Image to Fit Model
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = (
            midas_transforms.dpt_transform
            if model_type in [ModelType.DPT_LARGE, ModelType.DPT_HYBRID]
            else midas_transforms.small_transform
        )

    # Predict DepthMap for Each Frame
    def predict_depth(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img).to(self.device) # Transform Image and Send Tensor to Device

        with torch.no_grad(): # Non-Training Mode -> Don't Save Gradients
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate( # DepthMap Has to Fit Original Frame
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_raw = prediction.cpu().numpy() # Real Depth
        depth_norm = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

        return depth_colored, depth_raw

    # Read Images from Webcam
    def run_live(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found...")
            return

        print("Webcam opened (PRESS 'q' to stop)")
        previous_status = None # Save Previous Status to Avoid Unnecessary Prints

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Rectangle Size and Position
            h, w, _ = frame.shape
            roi = (w//2 - w//8, h//2 - h//8, w//4, h//4)

            depth_colored, depth_raw = self.predict_depth(frame) # Predict DepthMap for Whole Frame

            x, y, roi_w, roi_h = roi
            roi_depth = depth_raw[y:y+roi_h, x:x+roi_w] # Analyze Pixels in Rectangle

            threshold = 0.8 * depth_raw.max()
            danger = np.any(roi_depth > threshold) # Check if Pixel is Too Close

            # Only Print when Status Changes
            if danger and previous_status != "STOP":
                print(f"STOP! {time.time():.2f}")
                previous_status = "STOP"
            elif not danger and previous_status != "GO":
                print(f"GO! {time.time():.2f}")
                previous_status = "GO"

            # Draw Rectangle
            color = (0, 0, 255) if danger else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + roi_w, y + roi_h), color, 2)

            # Combine Original Frame and DepthMap
            depth_resized = cv2.resize(depth_colored, (w, h)) # Resize DepthMap to Match Frame Size
            combined = np.hstack((frame, depth_resized)) # Combine Horizontally

            cv2.imshow("Depth Detection", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()