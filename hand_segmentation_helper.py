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

# 为每种动物类型定义不同的色板
# Rabbit色板：蓝色系
rabbit_colors = [
    (255, 182, 193),  # 粉红
    (255, 218, 185),  # 桃子橘
    (255, 228, 196),  # 浅杏仁
    (255, 222, 173),  # 淡金色
    (255, 192, 203),  # 樱花粉
]

# Spider色板：红色系
spider_colors = [
    (178, 34, 34),     # 火焰红
    (75, 0, 130),      # 靛蓝
    (255, 140, 0),     # 暗橙色
    (47, 79, 79),      # 深石板灰
    (128, 0, 64),      # 暗莓红
]

# 存储每个具体角色的颜色
character_specific_colors = {}

# 为每种动物类型维护颜色索引
animal_color_index = {
    "rabbit": 0,
    "spider": 0
}

def get_color_for_character(character_name):
    """为角色获取或分配一个唯一的颜色，基于动物类型选择色板"""
    global character_specific_colors, animal_color_index
    
    # 如果角色已经有颜色，直接返回
    if character_name in character_specific_colors:
        return character_specific_colors[character_name]
    
    # 根据角色名称确定动物类型
    animal_type = "rabbit"  # 默认为rabbit
    character_lower = character_name.lower()
    
    if "spider" in character_lower:
        animal_type = "spider"
    
    # 根据动物类型选择色板
    if animal_type == "rabbit":
        colors = rabbit_colors
    else:  # spider
        colors = spider_colors
    
    # 获取当前动物类型的颜色索引
    color_idx = animal_color_index[animal_type]
    
    # 分配颜色
    if color_idx < len(colors):
        color = colors[color_idx]
        # 更新索引，供下一个同类型角色使用
        animal_color_index[animal_type] = (color_idx + 1) % len(colors)
    else:
        # 如果用完了所有颜色，生成一个随机颜色
        color = (
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256)
        )
    
    # 存储这个角色的颜色
    character_specific_colors[character_name] = color
    print(f"为角色 '{character_name}' 分配了新颜色 {color}，使用了{animal_type}色板")
    
    return color

def change_background_color():
    """Change the background color to the next one in the list"""
    global color_index, current_background_color
    color_index = (color_index + 1) % len(background_colors)
    current_background_color = background_colors[color_index]
    print(f"Changed background color to {current_background_color}")
    return current_background_color


def prepare_morphology_kernels():
    """準備形態學操作所需的核心"""
    kernels = {
        # 改用圆形或椭圆形结构元素，使边缘更圆润
        'dilation': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        'dilation_mask': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        'closing': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        'dilation2': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 710)),
        # 添加新的结构元素用于平滑处理
        'smoothing': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    }
    return kernels

def process_hand_segmentation(image, skeleton_binary, kernels, handedness=None, character_ids=None, landmarks=None):
    """處理手部分割，不再使用复杂掩码，直接绘制彩色骨骼
    
    参数:
        image: 输入图像
        skeleton_binary: 二值骨骼图像 (不再使用)
        kernels: 形态学操作的核心 (不再使用)
        handedness: 手部类型列表 (左手/右手)
        character_ids: 角色ID字典 {手部类型: 角色名称}
        landmarks: 手部关键点列表
    """
    global current_background_color, character_specific_colors
    
    # 创建纯色背景
    binary_output = np.ones_like(image, dtype=np.uint8)
    binary_output[:,:] = current_background_color
    
    # 如果没有landmarks或handedness，直接返回纯背景
    if not landmarks or not handedness or len(landmarks) == 0 or len(handedness) == 0:
        return binary_output
    
    # 定义MediaPipe手部关键点连接
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
    ]
    
    # 遍历每只手
    for i, (hand_type, hand_landmarks) in enumerate(zip(handedness, landmarks)):
        if i >= len(landmarks):
            continue
            
        # 获取角色ID
        character = character_ids.get(hand_type, "")
        
        # 为角色分配颜色
        if character:
            color = get_color_for_character(character)
            print(f"将角色 '{character}' 的手部骨骼颜色设置为 {color}")
        else:
            # 默认黑色
            color = (0, 0, 0)
        
        # 绘制手部关键点之间的连接线 - 增加线条粗细
        for connection in connections:
            start_idx, end_idx = connection
            
            # 确保索引在有效范围内
            if start_idx >= len(hand_landmarks) or end_idx >= len(hand_landmarks):
                continue
            
            # 获取关键点坐标
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            
            # 绘制线条 - 增加线条粗细到5
            cv2.line(
                binary_output, 
                start_point, 
                end_point, 
                color, 
                thickness=60  # 增加线条粗细
            )
        
        # 绘制关键点 - 也稍微增大
        for landmark in hand_landmarks:
            cv2.circle(
                binary_output, 
                landmark, 
                radius=6,  # 稍微增大关键点半径
                color=color, 
                thickness=-1  # 填充圆
            )
        
        # 移除角色标签文字
    
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