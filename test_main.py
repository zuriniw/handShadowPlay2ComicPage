import mediapipe as mp
import cv2
import numpy as np
import time
import os
import sys
import traceback

# Import MediaPipe Tasks modules
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Import our gesture recognition module
import gesture_recognition
import hand_segmentation_helper as helper
import keyframe_tracker

# Override the default keyframe tracker with our test file
def create_test_tracker():
    return keyframe_tracker.KeyframeTracker("test_keyframes.json")

# Replace the global tracker with our test tracker
keyframe_tracker._keyframe_tracker = create_test_tracker()

# Print the count of keyframes
test_tracker = keyframe_tracker.get_keyframe_tracker()
print(f"Test tracker has {len(test_tracker.keyframes)} keyframes")

# Trigger keyframe limit check to verify termination
should_terminate = test_tracker.check_keyframe_limit()
print(f"Should terminate: {should_terminate}")

# Exit without running the main camera loop
print("Test completed successfully")
sys.exit(0) 