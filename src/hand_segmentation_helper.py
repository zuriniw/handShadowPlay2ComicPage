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
    (141, 182, 182),  # 青石灰
    (119, 160, 160),  # 雾蓝绿
    (128, 179, 174),  # 冷调青
    (144, 180, 123),  # 苔绿色
    (174, 204, 110),  # 芥末绿
    (184, 216, 144),  # 幼芽绿
    (198, 224, 140),  # 草原浅绿
]
# Index to keep track of which color to use next
color_index = 0

# Define different color palettes for each animal type
# Rabbit palette: blue-based
rabbit_colors = [
    (255, 182, 193),  # 粉红
    (255, 218, 185),  # 桃子橘
    (255, 228, 196),  # 浅杏仁
    (255, 222, 173),  # 淡金色
    (255, 192, 203),  # 樱花粉
]

# Spider palette: red-based
spider_colors = [
    (178, 34, 34),     # 火焰红
    (75, 0, 130),      # 靛蓝
    (255, 140, 0),     # 暗橙色
    (47, 79, 79),      # 深石板灰
    (128, 0, 64),      # 暗莓红
]

# Store the color for each specific character
character_specific_colors = {}

# Maintain a color index for each animal type
animal_color_index = {
    "rabbit": 0,
    "spider": 0
}

def get_color_for_character(character_name):
    """Get or assign a unique color for the character based on the animal type"""
    global character_specific_colors, animal_color_index
    
    # If the character already has a color, return it directly
    if character_name in character_specific_colors:
        return character_specific_colors[character_name]
    
    # Determine the animal type based on the character name
    animal_type = "rabbit"  # Default to rabbit
    character_lower = character_name.lower()
    
    if "spider" in character_lower:
        animal_type = "spider"
    
    # Select the color palette based on the animal type
    if animal_type == "rabbit":
        colors = rabbit_colors
    else:  # spider
        colors = spider_colors
    
    # Get the current color index for the animal type
    color_idx = animal_color_index[animal_type]
    
    # Assign a color
    if color_idx < len(colors):
        color = colors[color_idx]
        # Update the index, for use by the next character of the same type
        animal_color_index[animal_type] = (color_idx + 1) % len(colors)
    else:
        # If all colors are used, generate a random color
        color = (
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256)
        )
    
    # Store the color for this character
    character_specific_colors[character_name] = color
    print(f"Assigned new color {color} to character '{character_name}', using {animal_type} palette")
    
    return color

def change_background_color():
    """Change the background color to the next one in the list"""
    global color_index, current_background_color
    color_index = (color_index + 1) % len(background_colors)
    current_background_color = background_colors[color_index]
    print(f"Changed background color to {current_background_color}")
    return current_background_color

def process_hand_segmentation(image, handedness=None, character_ids=None, landmarks=None):
    global current_background_color, character_specific_colors
    
    # Create a solid color background
    binary_output = np.ones_like(image, dtype=np.uint8)
    binary_output[:,:] = current_background_color
    
    # If there are no landmarks or handedness, return the pure background
    if not landmarks or not handedness or len(landmarks) == 0 or len(handedness) == 0:
        return binary_output
    
    # Define MediaPipe hand landmark connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
    ]
    
    # Iterate through each hand
    for i, (hand_type, hand_landmarks) in enumerate(zip(handedness, landmarks)):
        if i >= len(landmarks):
            continue

        character = character_ids.get(hand_type, "")
 
        if character:
            color = get_color_for_character(character)
            print(f"Setting the hand bone color for character '{character}' to {color}")
        else:
            color = (0, 0, 0)

        for connection in connections:
            start_idx, end_idx = connection

            if start_idx >= len(hand_landmarks) or end_idx >= len(hand_landmarks):
                continue

            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]

            cv2.line(
                binary_output, 
                start_point, 
                end_point, 
                color, 
                thickness=60  # Increase line thickness
            )

        for landmark in hand_landmarks:
            cv2.circle(
                binary_output, 
                landmark, 
                radius=6, 
                color=color, 
                thickness=-1
            )
    
    return binary_output

def create_info_panel(image_width, processed_results):
    """Create an info panel to display the animal characters in the current scene and all characters that have appeared"""
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
    """Annotate the image with hand type and gesture"""
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
    """Measure the distance between two characters and annotate it on the image"""
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