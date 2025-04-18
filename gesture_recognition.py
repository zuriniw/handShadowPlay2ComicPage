import mediapipe as mp 
import cv2
import numpy as np
import os
import sys
import traceback
import hashlib

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Global variable to store current gesture results
current_gesture_results = {"Left": None, "Right": None}  # "Left" and "Right" refer to handedness, not screen position
# Sticky labels for each hand
sticky_labels = {"Left": None, "Right": None}
# Character identity mapping
character_ids = {}
character_counter = {"snake": 1, "rabbit": 1}  # Start from 1 instead of 0
# Track assigned identities even after detachment
identity_history = {"Left": None, "Right": None}
# Keep a record of all characters that have ever appeared
all_characters_history = set()

def process_result(result, output_image, timestamp_ms):
    processed_results = {"Left": None, "Right": None}  # Handedness-based
    seen_hands = set()

    if result.gestures:
        for i, gesture in enumerate(result.gestures):
            top_gesture = gesture[0]
            handedness = result.handedness[i][0].category_name  # "Right" or "Left"
            flipped_handedness = "Right" if handedness == "Left" else "Left"
            seen_hands.add(flipped_handedness)

            gesture_name = top_gesture.category_name.lower()

            if gesture_name in ["closed_fist", "fist"]:
                gesture_name = "snake"
            elif gesture_name in ["victory", "peace"]:
                gesture_name = "rabbit"
            elif gesture_name in ["open_palm", "open_hand"]:
                gesture_name = "terminate"
            else:
                gesture_name = None  # Ignore all other gestures

            # Stickiness logic with independent identity tracking
            if sticky_labels[flipped_handedness] is None and gesture_name in ["snake", "rabbit"]:
                sticky_labels[flipped_handedness] = gesture_name
                character_id = f"{gesture_name} {character_counter[gesture_name]}"
                character_ids[flipped_handedness] = character_id
                identity_history[flipped_handedness] = character_id
                # Add to the complete history of all characters
                global all_characters_history
                all_characters_history.add(character_id)
                character_counter[gesture_name] += 1

    # Update only based on sticky labels
    for hand in ["Left", "Right"]:
        if hand in seen_hands:
            processed_results[hand] = sticky_labels[hand]
        else:
            # Hand is not visible anymore, clear its sticky label
            sticky_labels[hand] = None
            processed_results[hand] = None
            # Remove from current character mapping but keep the identity history
            character_ids.pop(hand, None)  # Don't reset identity history

    global current_gesture_results
    current_gesture_results = processed_results

def load_model():
    local_model_path = os.path.join(".", "gesture_recognizer.task")
    model_bytes = None

    if os.path.exists(local_model_path):
        with open(local_model_path, 'rb') as f:
            model_bytes = f.read()
        return model_bytes
    else:
        print(f"❌ 找不到本地模型檔案: {local_model_path}")
        return None

def create_recognizer():
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_bytes = load_model()
    if not model_bytes or len(model_bytes) < 1000:
        print("❌ 模型資料無效或大小不足")
        return None

    model_hash = hashlib.md5(model_bytes).hexdigest()
    print(f"✅ 模型檔案 MD5: {model_hash}")

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=process_result,
        num_hands=2
    )

    try:
        recognizer = GestureRecognizer.create_from_options(options)
        print("✅ 成功建立手勢識別器")
        return recognizer
    except Exception as e:
        print(f"❌ 無法創建手勢識別器: {str(e)}")
        traceback.print_exc()
        return None

def get_current_results():
    global current_gesture_results
    return current_gesture_results

def get_character_labels():
    global character_ids
    return character_ids

def get_identity_history():
    global identity_history
    return identity_history

def get_all_characters_history():
    """Get a set of all animal characters that have ever appeared"""
    global all_characters_history
    return all_characters_history

def reset_results():
    global current_gesture_results, sticky_labels, character_ids, character_counter, identity_history, all_characters_history
    current_gesture_results = {"Left": None, "Right": None}
    sticky_labels = {"Left": None, "Right": None}
    character_ids = {}
    identity_history = {"Left": None, "Right": None}
    character_counter = {"snake": 1, "rabbit": 1}
    all_characters_history = set()

def draw_labels_on_frame(frame, result):
    """將動物標籤顯示在畫面上對應的手部位置"""
    if not result.hand_landmarks:
        return frame

    image_width = frame.shape[1]
    labels = get_character_labels()
    identity_map = get_identity_history()

    for i, landmarks in enumerate(result.hand_landmarks):
        # Handle both MediaPipe Hands format and our custom format
        try:
            # First try the original format used in process_result
            handedness = result.handedness[i][0].category_name
        except (TypeError, IndexError):
            try:
                # Then try the MediaPipe Hands format
                handedness = result.handedness[i].classification[0].label
            except (AttributeError, IndexError):
                # Skip if handedness cannot be determined
                continue
                
        flipped_handedness = "Right" if handedness == "Left" else "Left"
        label = labels.get(flipped_handedness, identity_map.get(flipped_handedness, ""))

        # Get wrist position depending on landmarks type
        try:
            # Check if landmarks is a NormalizedLandmarkList (MediaPipe format)
            wrist = landmarks.landmark[0]  # This is for MediaPipe's native format
            x = int(frame.shape[1] - wrist.x * frame.shape[1])  # flip x
            y = int(wrist.y * frame.shape[0])
        except AttributeError:
            # If not, it might be our converted landmarks list of tuples
            try:
                wrist_px, wrist_py = landmarks[0]  # This is for our converted format
                x = int(frame.shape[1] - wrist_px)  # flip x
                y = wrist_py
            except (TypeError, IndexError):
                # Skip if wrist position cannot be determined
                continue

        # Only draw label if one exists
        if label:
            cv2.putText(frame, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

    return frame
