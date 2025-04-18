import cv2
import numpy as np

def prepare_morphology_kernels():
    """準備形態學操作所需的核心"""
    kernels = {
        'dilation': cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)),
        'dilation_mask': cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        'closing': cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)),
        'dilation2': cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    }
    return kernels

def process_hand_segmentation(image, skeleton_binary, kernels):
    """處理手部分割"""
    # Stage 2: Get hand zone
    skeleton_binary = cv2.dilate(skeleton_binary, kernels['dilation_mask'], iterations=1)
    mask_s2 = cv2.dilate(skeleton_binary, kernels['dilation'], iterations=2)
    
    # Convert image to LAB for hand segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Get boolean mask of interest pixels
    mask_landmarks = skeleton_binary == 255
    # Get hue, saturation and value values in previously selected pixels
    masked_hsv = np.where(mask_landmarks != 0)
    
    # Create binary output
    binary_output = np.ones_like(image) * 255  # Create white background
    
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
            binary_output[mask_hsv[:,:,0] == 255] = 0  
    
    return binary_output

def create_info_panel(image_width, processed_results):
    """創建資訊面板，显示当前场景中的动物角色和历史上出现过的所有角色"""
    # Increase panel height to accommodate the character history
    info_panel_height = 180
    info_panel = np.ones((info_panel_height, image_width, 3), dtype=np.uint8) * 255
    
    # Import character ID information from gesture recognition
    import gesture_recognition

    # Get current characters and complete history of all characters
    current_characters = gesture_recognition.get_character_labels()
    all_characters = gesture_recognition.get_all_characters_history()
    
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
            
            # Get character ID for this hand if available
            char_id = character_ids.get(hand_type, identity_history.get(hand_type, ""))
            
            # Display only the character ID if available, otherwise show hand type
            if char_id:
                # Just show the character ID
                annotation_text = char_id
                text_color = (0, 255, 255)  # Yellow for character labels
            elif gesture_text:
                # If there's a gesture but no character ID
                annotation_text = f"{hand_type}: {gesture_text}"
                text_color = (0, 255, 0)  # Green for recognized gestures
            else:
                # If no gesture is recognized
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