import cv2
import numpy as np
import math

# Add keyframe_tracker import
import keyframe_tracker

# Global variable to track current background color
# Format: (B, G, R)
current_background_color = (255, 255, 255)  # Start with white background
# List of background colors to cycle through - use more distinct colors
background_colors = [
    (255, 255, 255),  # White
    (180, 180, 255),  # Light Red
    (180, 255, 180),  # Light Green
    (255, 180, 180),  # Light Blue
    (180, 255, 255),  # Light Yellow
    (255, 180, 255),  # Light Purple
    (255, 255, 180),  # Light Cyan
]
# Index to keep track of which color to use next
color_index = 0

def prepare_morphology_kernels():
    """準備形態學操作所需的核心"""
    kernels = {
        'dilation': cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)),
        'dilation_mask': cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        'closing': cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)),
        'dilation2': cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    }
    return kernels

def change_background_color():
    """Change the background color to the next one in the list"""
    global color_index, current_background_color
    color_index = (color_index + 1) % len(background_colors)
    current_background_color = background_colors[color_index]
    print(f"Changed background color to {current_background_color}")
    return current_background_color

def process_hand_segmentation(image, skeleton_binary, kernels):
    """處理手部分割"""
    global current_background_color
    
    # Stage 2: Get hand zone
    skeleton_binary = cv2.dilate(skeleton_binary, kernels['dilation_mask'], iterations=1)
    mask_s2 = cv2.dilate(skeleton_binary, kernels['dilation'], iterations=2)
    
    # Convert image to LAB for hand segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Get boolean mask of interest pixels
    mask_landmarks = skeleton_binary == 255
    # Get hue, saturation and value values in previously selected pixels
    masked_hsv = np.where(mask_landmarks != 0)
    
    # Create binary output with the current background color
    # Need to create the array with proper shape first
    binary_output = np.ones_like(image, dtype=np.uint8)
    # Apply the current background color to all pixels
    binary_output[:,:] = current_background_color
    
    # Make sure we have detected hands before proceeding
    if len(masked_hsv[0]) > 0:
        masked_hue = hsv_image[:, :, 0][masked_hsv]
        masked_sat = hsv_image[:, :, 1][masked_hsv]
        masked_val = hsv_image[:, :, 2][masked_hsv]

        n_samples = 150
        if len(masked_sat) > 0 and len(masked_val) > 0:
            q1s, q3s = np.percentile(np.random.choice(masked_sat, min(n_samples, len(masked_sat))), [25, 75])
            q1v, q3v = np.percentile(np.random.choice(masked_val, min(n_samples, len(masked_val))), [25, 75])

            factors = 0.025
            factorv = 0.025
            mask_sat = cv2.inRange(hsv_image[:, :, 1], int(q1s * (1 - factors)), int(q3s * (1 + factors)))
            mask_val = cv2.inRange(hsv_image[:, :, 2], int(q1v * (1 - factorv)), int(q3v * (1 + factorv)))
            mask_hsv = (mask_sat & mask_val) | skeleton_binary
            mask_hsv = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2RGB)

            mask_hsv |= cv2.cvtColor(skeleton_binary, cv2.COLOR_GRAY2RGB)

            mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernels['closing'], iterations=1)
            mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernels['closing'], iterations=1)
            mask_hsv = cv2.dilate(mask_hsv, kernels['dilation2'], iterations=1)
            mask_hsv = mask_hsv & cv2.cvtColor(mask_s2, cv2.COLOR_GRAY2RGB)

            # Set hand area to black in binary output
            binary_output[mask_hsv[:,:,0] == 255] = (0, 0, 0)
    
    return binary_output

def create_info_panel(image_width, processed_results):
    """創建資訊面板，显示当前场景中的动物角色和历史上出现过的所有角色"""
    # Increase panel height to accommodate the character history and keyframe info
    info_panel_height = 210
    info_panel = np.ones((info_panel_height, image_width, 3), dtype=np.uint8) * 255
    
    # Import character ID information from gesture recognition
    import gesture_recognition

    # Get current characters and complete history of all characters
    current_characters = gesture_recognition.get_character_labels()
    all_characters = gesture_recognition.get_all_characters_history()
    
    # Get the keyframe tracker from keyframe_tracker module
    kf_tracker = keyframe_tracker.get_keyframe_tracker()
    
    # Display current gesture recognition results with smaller text
    if not any(processed_results.values()):
        cv2.putText(info_panel, "No gestures detected", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    else:
        # Left hand info
        left_text = f"Left: {processed_results['Left'] if processed_results['Left'] else 'None'}"
        cv2.putText(info_panel, left_text, 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 0, 255) if not processed_results['Left'] else (0, 128, 0), 1)
        
        # Right hand info
        right_text = f"Right: {processed_results['Right'] if processed_results['Right'] else 'None'}"
        cv2.putText(info_panel, right_text, 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 0, 255) if not processed_results['Right'] else (0, 128, 0), 1)
    
    # Display current characters in scene with abbreviated label
    cv2.putText(info_panel, "CC:", 
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    current_char_text = ", ".join([char_id for char_id in current_characters.values() if char_id]) or "None"
    cv2.putText(info_panel, current_char_text, 
                (70, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 1)
    
    # Display all characters that have appeared with abbreviated label
    cv2.putText(info_panel, "AC:", 
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Use the comprehensive history of all characters
    all_char_list = sorted(all_characters)
    
    # Handle case where the list might be too long for one line
    if len(all_char_list) > 0:
        all_char_text = ", ".join(all_char_list)
        
        # Check if text might be too long and split if needed
        if len(all_char_text) > 50:
            line1 = ", ".join(all_char_list[:len(all_char_list)//2])
            line2 = ", ".join(all_char_list[len(all_char_list)//2:])
            
            cv2.putText(info_panel, line1, 
                       (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 0), 1)
            cv2.putText(info_panel, line2, 
                       (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 0), 1)
        else:
            cv2.putText(info_panel, all_char_text, 
                       (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 0), 1)
    else:
        cv2.putText(info_panel, "None", 
                   (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 0), 1)
    
    # Display latest keyframe if available
    cv2.putText(info_panel, "KF:", 
                (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    if kf_tracker.keyframes:
        latest_keyframe = kf_tracker.keyframes[-1]
        # Use timestamp instead of time if time is not available
        timestamp_str = latest_keyframe.get('time', str(latest_keyframe.get('timestamp', 0)))
        keyframe_text = f"{timestamp_str} - {latest_keyframe['name']}"
        cv2.putText(info_panel, keyframe_text, 
                   (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
    else:
        cv2.putText(info_panel, "None", 
                   (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
    
    return info_panel

def annotate_image(image, handedness, landmarks, processed_results):
    """在圖像上標注手部類型和手勢"""
    result_image = image.copy()
    
    # Import gesture recognition module to get character IDs
    import gesture_recognition
    character_ids = gesture_recognition.get_character_labels()
    identity_history = gesture_recognition.get_identity_history()
    
    if handedness and landmarks:
        for i, (hand_type, hand_landmarks) in enumerate(zip(handedness, landmarks)):
            # Use wrist position for text
            text_pos = hand_landmarks[0]
            gesture_text = processed_results.get(hand_type, None)
            
            # Display based on current recognized gesture state
            if hand_type in character_ids and character_ids[hand_type]:
                # Hand has a current character ID
                annotation_text = character_ids[hand_type]
                text_color = (0, 255, 255)  # Yellow for character labels
            elif gesture_text:
                # Hand has a recognized gesture but no character ID yet
                annotation_text = f"{hand_type}: {gesture_text}"
                text_color = (0, 255, 0)  # Green for recognized gestures
            else:
                # Hand with no recognized gesture - just show handedness
                annotation_text = f"{hand_type}"
                text_color = (0, 0, 255)  # Red for unrecognized hands
            
            # Draw text with smaller font size and thinner line
            cv2.putText(result_image, 
                        annotation_text, 
                        (text_pos[0], text_pos[1] - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7,  # Smaller text
                        text_color, 
                        1)  # Thinner line
    
    return result_image

def measure_character_distance(image, handedness, landmarks, character_ids):
    """测量两个角色之间的距离并在图像上进行标注"""
    # Make a copy of the input image
    distance_image = image.copy()
    
    # If we don't have enough hands or character IDs, return the original image
    if len(handedness) < 2 or len(landmarks) < 2 or len(character_ids) < 2:
        return distance_image
        
    # Check if we have at least two characters with IDs
    characters_with_ids = []
    hand_landmarks = []
    hand_types = []
    
    # Collect hands that have character IDs (use only current character_ids, not history)
    for i, hand_type in enumerate(handedness):
        if i < len(landmarks) and hand_type in character_ids and character_ids[hand_type]:
            characters_with_ids.append(character_ids[hand_type])
            hand_landmarks.append(landmarks[i])
            hand_types.append(hand_type)
    
    # We need at least two hands with character IDs to measure distance
    if len(characters_with_ids) < 2:
        return distance_image
    
    # Get the first two hands with character IDs
    char1 = characters_with_ids[0]
    char2 = characters_with_ids[1]
    hand1_wrist = hand_landmarks[0][0]  # First point is wrist
    hand2_wrist = hand_landmarks[1][0]  # First point is wrist
    
    # Calculate Euclidean distance between the two wrists
    distance = math.sqrt((hand1_wrist[0] - hand2_wrist[0])**2 + 
                        (hand1_wrist[1] - hand2_wrist[1])**2)
    
    # Convert to pixel distance
    distance_pixels = int(distance)
    
    # Draw a line connecting the two wrists
    cv2.line(distance_image, hand1_wrist, hand2_wrist, (0, 255, 255), 2)
    
    # Calculate midpoint for text placement
    mid_x = (hand1_wrist[0] + hand2_wrist[0]) // 2
    mid_y = (hand1_wrist[1] + hand2_wrist[1]) // 2
    
    # Draw distance text at the midpoint
    distance_text = f"Distance: {distance_pixels}px"
    cv2.putText(distance_image, distance_text, 
                (mid_x + 10, mid_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
    
    # Draw character interaction text
    interaction_text = f"{char1} <-> {char2}"
    cv2.putText(distance_image, interaction_text, 
                (mid_x + 10, mid_y - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
    
    return distance_image 