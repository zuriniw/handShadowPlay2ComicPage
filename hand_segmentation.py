import mediapipe as mp
import cv2
import time
import os
import tqdm
import numpy as np
import hands_connections
from sklearn.metrics import jaccard_score


def extract_hands(image):
    """
    :return: Tuple of:
    Black image with the skeleton drawn,
    Array with x, y position of all landmarks,
    Message with 21 * 4 + 2 elements where the two first elements can be "L", "R" or -100.
    These two first elements signal the order of the handedness in the following elements. The next
    84 elements are integers signaling the x,y coordinates in screen space (pixels) of each one
    of the 21 landmark points of each hand. For example, if the message is R, L, x1, y1, ...
    it means that the first 42 integers will correspond to the right hand while the last 42 will be
    the positions of the left hand landmarks.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    black_canvas = np.zeros_like(image, dtype=np.uint8)
    out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # https://google.github.io/mediapipe/solutions/hands.html#output
    if results.multi_hand_landmarks:
        # Se almacena uno por cada mano: len(results.multi_hand_landmarks) = 1 o 2
        for hi, (handedness, hand) in enumerate(zip(results.multi_handedness, results.multi_hand_landmarks)):
            mp_drawing.draw_landmarks(
                black_canvas,
                hand,
                hands_connections.HAND_CONNECTIONS,
                mp_drawing_styles.get_custom_hand_connections_style())
            mp_drawing.draw_landmarks(out, hand, hands_connections.HAND_CONNECTIONS, mp_drawing_styles.get_custom_hand_connections_style())
    return black_canvas, out


def binaryMaskIOU(mask1, mask2):   # From the question.
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and( mask1==1,  mask2==1 ))
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou


def list_RGB2BGR(a):
    a[0], a[2] = a[2], a[0]
    return a


def draw_mask(image, mask, draw_zero=False, alpha=1.0, colors_dict=None):
    """
    colors_dict overwrites alpha if colors have 4 values and draw zero is a key "0" is in the dict.
    alpha can be overwritten by colors with 4 values.
    Add annotations
    """
    img = image.copy()
    # Case 1: Array of masks

    # Case 2: Image with a different pixel value for each mask
    assert image.shape
    unique_values = np.unique(mask)
    if True:
        # Convert colors to BGR and convert them to np.array
        colors_dict = {k: np.array(list_RGB2BGR(v)) for k, v in colors_dict.items()}
        if "0" in colors_dict:
            draw_zero = True
    for i, color in colors_dict.items():
        if draw_zero or i != 0:
            img_mask = mask[:, :, 0] == int(i)
            i_alpha = alpha
            if len(color) == 4:
                i_alpha = color[-1] / 255
            color = img[img_mask] * (1 - i_alpha) + color[:3] * i_alpha
            img[img_mask] = color
    return img


if __name__ == "__main__":
    # Mediapipe settings
    mp_hands = mp.solutions.hands
    import custom_drawing
    mp_drawing = custom_drawing
    mp_drawing_styles = custom_drawing

    # 初始化网络摄像头
    cap = cv2.VideoCapture(0)
    # 设置较低的帧率，比如15fps
    cap.set(cv2.CAP_PROP_FPS, 15)

    # 准备形态学操作的kernel
    kernel_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    kernel_dilation_mask = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    kernel_dilation2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    # Open mediapipe process
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=2, static_image_mode=False) as hands:
        print("Mediapipe initialized, starting webcam capture...")
        
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Stage 1: Get landmark mask from Mediapipe
            skeleton, out = extract_hands(image)
            skeleton_gray = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
            _, skeleton_binary = cv2.threshold(skeleton_gray, 1, 255, cv2.THRESH_BINARY)

            # Stage 2: Get hand zone
            skeleton_binary = cv2.dilate(skeleton_binary, kernel_dilation_mask, iterations=1)
            mask_s2 = cv2.dilate(skeleton_binary, kernel_dilation, iterations=2)

            try:
                # Convert image to hsv
                hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

                # Get boolean mask of interest pixels
                mask_landmarks = skeleton_binary == 255
                # Get hue, saturation and value values in previously selected pixels
                masked_hsv = np.where(mask_landmarks != 0)
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

                    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel_closing, iterations=1)
                    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel_closing, iterations=1)
                    mask_hsv = cv2.dilate(mask_hsv, kernel_dilation2, iterations=1)
                    mask_hsv = mask_hsv & cv2.cvtColor(mask_s2, cv2.COLOR_GRAY2RGB)

                    # 显示结果
                    # 将手部区域设为黑色(0)，其余区域设为白色(255)
                    binary_output = np.ones_like(image) * 255  # 创建白色背景
                    binary_output[mask_hsv[:,:,0] == 255] = 0  # 手部区域设为黑色
                    
                    # 显示处理后的图像
                    cv2.imshow("Hand Detection", binary_output)
                    cv2.imshow("Mediapipe Output", out)

            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
