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

def extract_hands_landmarks(image, hands):
    """
    Extract hand landmarks and information from an image using regular MediaPipe Hands
    """
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)

        black_canvas = np.zeros_like(image, dtype=np.uint8)
        out = image.copy()
        
        landmarks = []
        handedness_list = []
        
        if results.multi_hand_landmarks:
            for hi, (handedness, hand) in enumerate(zip(results.multi_handedness, results.multi_hand_landmarks)):
                # Store handedness (left or right hand)
                handedness_info = handedness.classification[0].label
                handedness_list.append(handedness_info)
                
                # Store landmarks as pixel coordinates
                hand_landmarks = []
                for landmark in hand.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    h, w, _ = image.shape
                    px, py = int(landmark.x * w), int(landmark.y * h)
                    hand_landmarks.append((px, py))
                landmarks.append(hand_landmarks)
                
                mp_drawing.draw_landmarks(
                    black_canvas,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                mp_drawing.draw_landmarks(
                    out, 
                    hand, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        return black_canvas, out, landmarks, handedness_list, results
    except Exception as e:
        print(f"Error in extract_hands_landmarks: {str(e)}")
        traceback.print_exc()
        return np.zeros_like(image), image.copy(), [], [], None


if __name__ == "__main__":
    try:
        # MediaPipe regular hands settings (for visualization)
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize webcam with error handling
        # print("正在嘗試啟用攝影機...")
        cap = cv2.VideoCapture(0)
        # Set lower framerate for better performance
        cap.set(cv2.CAP_PROP_FPS, 15)
        if not cap.isOpened():
            # print("❌ 無法啟用攝影機，請確認是否被其他程式佔用")
            sys.exit(1)
            
        # print("✅ 成功啟用攝影機")

        # Prepare kernels for morphological operations using helper
        kernels = helper.prepare_morphology_kernels()

        # print("正在準備 MediaPipe 模型...")
        
        try:
            # Create gesture recognizer using our module
            recognizer = gesture_recognition.create_recognizer()
            if recognizer is None:
                # print("❌ 無法創建手勢識別器")
                if cap is not None and cap.isOpened():
                    cap.release()
                sys.exit(1)
            
            # Open regular MediaPipe Hands for visualization and hand extraction
            # print("正在啟動手部識別系統...")
            
            with mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_hands=2, 
                static_image_mode=False) as hands:
                
                # print("✅ MediaPipe 初始化完成，開始視訊擷取...")
                # print("按下 'q' 退出程式，按下 'd' 切換除錯視覺化")
                
                # Debug flag for visualizing finger states
                debug_visualization = False
                
                # Check camera is still accessible
                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    # print("❌ 無法從攝影機讀取畫面")
                    recognizer.close()
                    sys.exit(1)
                
                # Store previous keyframe count to detect new keyframes
                previous_keyframe_count = 0
                
                while cap.isOpened():
                    try:
                        ret, image = cap.read()
                        if not ret:
                            # print("❌ 無法擷取畫面")
                            break
                            
                        # Flip the input image horizontally
                        image = cv2.flip(image, 1)

                        # Get current timestamp for the frame
                        timestamp_ms = int(time.time() * 1000)
                        
                        # Stage 1: Get landmark mask from regular MediaPipe Hands (for visualization)
                        skeleton, out, landmarks, handedness, results = extract_hands_landmarks(image, hands)
                        skeleton_gray = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
                        _, skeleton_binary = cv2.threshold(skeleton_gray, 1, 255, cv2.THRESH_BINARY)
                        
                        # Convert image to MediaPipe Image format and process with the gesture recognizer
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        
                        # Send the image to the recognizer with timestamp
                        recognizer.recognize_async(mp_image, timestamp_ms)
                        
                        # Get the processed results from our module
                        processed_results = gesture_recognition.get_current_results()

                        # Get character IDs and keyframe tracker before processing the hands
                        # to ensure we have the latest keyframe state
                        character_ids = gesture_recognition.get_character_labels()
                        kf_tracker = keyframe_tracker.get_keyframe_tracker()
                        all_characters_history = gesture_recognition.get_all_characters_history()

                        # Check if a new keyframe has been added
                        current_keyframe_count = len(kf_tracker.keyframes)
                        if current_keyframe_count > previous_keyframe_count:
                            # A new keyframe was added, change the background color
                            helper.change_background_color()
                            previous_keyframe_count = current_keyframe_count
                            print(f"Keyframe count increased to {current_keyframe_count}, changed background color")

                        # Process hand segmentation using helper - pass handedness and character IDs
                        # This will color each recognized character with a unique color, while unrecognized hands remain the same color as the background
                        binary_output = helper.process_hand_segmentation(image, skeleton_binary, kernels, handedness, character_ids, landmarks)
                        
                        # Create info panel using helper
                        info_panel = helper.create_info_panel(image.shape[1], processed_results)
                        
                        # Annotate the result image with gesture information
                        result_image = helper.annotate_image(image, handedness, landmarks, processed_results)
                        
                        # Verify keyframes file periodically
                        if time.time() % 10 < 1:  # Roughly every 10 seconds
                            kf_tracker.verify_keyframes_file()

                        # Check if there are any new characters to announce
                        new_characters = gesture_recognition.get_new_characters_to_announce()
                        if new_characters:
                            # Force keyframe creation for each new character
                            current_chars = set(character_ids.values())
                            for new_char in new_characters:
                                # Use the force method for add character events too
                                print(f"Adding keyframe for new character: {new_char}")
                                kf_tracker.add_keyframe_force(
                                    f"add new character {new_char}",
                                    current_chars,
                                    all_characters_history
                                )
                                print(f"Keyframe added for new character: {new_char}")

                        # Check if there are any characters that quit
                        quit_characters = gesture_recognition.get_characters_that_quit()
                        if quit_characters:
                            # Force keyframe creation for each character that quit
                            current_chars = set(character_ids.values())
                            for quit_char in quit_characters:
                                # Ensure we properly notify when adding quit keyframes
                                print(f"Adding keyframe for quit character: {quit_char}")
                                # Add the keyframe and bypass normal time restrictions
                                kf_tracker.add_keyframe_force(
                                    f"quit character {quit_char}",
                                    current_chars,
                                    all_characters_history
                                )
                                print(f"Keyframe added for quit character: {quit_char}")
                        
                        # Check for new keyframes
                        any_hands_visible = len(handedness) > 0
                        kf_tracker.check_terminate(any_hands_visible, all_characters_history)
                        kf_tracker.check_character_changes(character_ids, all_characters_history)
                        kf_tracker.check_distance(character_ids, handedness, landmarks, all_characters_history)
                        
                        # Update keyframe count if it changed during processing
                        new_keyframe_count = len(kf_tracker.keyframes)
                        if new_keyframe_count > current_keyframe_count:
                            helper.change_background_color()
                            previous_keyframe_count = new_keyframe_count
                            print(f"Keyframe count increased to {new_keyframe_count} during processing, changed background color")
                        
                        # Check if we've reached the 6 keyframe limit
                        if kf_tracker.check_keyframe_limit():
                            # Instead of breaking, just set a flag to show black screen
                            # but continue running the program
                            pass
                        
                        # Measure and display distance between characters if available
                        result_image = helper.measure_character_distance(result_image, handedness, landmarks, character_ids)
                        
                        # Display debug visualization if enabled
                        if debug_visualization and landmarks:
                            result_image = gesture_recognition.visualize_finger_state(result_image, landmarks)
                        
                        # Combine info panel and result image
                        combined_image = np.vstack([result_image, info_panel])
                        
                        # Since input image is already flipped, we don't need to flip outputs again
                        # Display processed images
                        # If keyframe limit reached, show black screen for Hand Detection
                        if kf_tracker.is_limit_reached():
                            # Create a black image with same dimensions as binary_output
                            black_screen = np.zeros_like(binary_output)
                            cv2.imshow("Hand Detection", black_screen)
                        else:
                            cv2.imshow("Hand Detection", binary_output)
                        
                        cv2.imshow("MediaPipe Output", out)
                        cv2.imshow("Gesture Recognition", combined_image)

                    except Exception as e:
                        # print(f"❌ 處理畫面時發生錯誤: {str(e)}")
                        traceback.print_exc()
                        continue

                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):  # Press 'q' to quit
                        break
                    elif key == ord('d'):  # Press 'd' to toggle debug visualization
                        debug_visualization = not debug_visualization
                        # print(f"Debug visualization: {'ON' if debug_visualization else 'OFF'}")
                    elif key == ord('r'):  # Press 'r' to reset keyframes
                        kf_tracker.reset_keyframes()
                        gesture_recognition.reset_character_tracking()
                        # Reset background color index when resetting keyframes
                        helper.color_index = 0
                        helper.current_background_color = helper.background_colors[0]
                        previous_keyframe_count = 0
                        print("Keyframes reset. Press 'q' to quit.")

        except Exception as e:
            # print(f"❌ MediaPipe 模型載入或執行失敗: {str(e)}")
            traceback.print_exc()
            if "cap" in locals() and cap is not None and cap.isOpened():
                cap.release()
            sys.exit(1)
            
        finally:
            # Clean up resources
            # print("正在釋放資源...")
            # Make sure cap is valid before releasing
            if cap is not None and cap.isOpened():
                cap.release()
            # Close the recognizer
            if 'recognizer' in locals() and recognizer is not None:
                recognizer.close()
            cv2.destroyAllWindows()
            #       print("程式結束")
            
    except Exception as e:
        # print(f"❌ 程式啟動時發生嚴重錯誤: {str(e)}")
        traceback.print_exc()
        sys.exit(1)