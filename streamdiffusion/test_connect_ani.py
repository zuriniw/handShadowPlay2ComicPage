import os
import sys
import time
import threading
from multiprocessing import Process, Queue, get_context
from multiprocessing.connection import Connection
from typing import List, Literal, Dict, Optional
import torch
import PIL.Image
from streamdiffusion.image_utils import pil2tensor
import mss
import fire
import tkinter as tk
import win32gui
import win32con

import socket
import threading

latest_signal = None  # Global variable to store the latest signal
lock = threading.Lock()  # Lock for thread-safe access to latest_signal
prompt_updated = threading.Event()  # Event to track when prompt is updated

def find_window_by_title(title):
    def callback(hwnd, result):
        if win32gui.IsWindowVisible(hwnd) and title.lower() in win32gui.GetWindowText(hwnd).lower():
            result.append(hwnd)
    result = []
    win32gui.EnumWindows(callback, result)
    return result[0] if result else None

def get_window_rect(hwnd):
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return {
        "left": left + 8,
        "top": top + 30,
        "width": right - left - 16,
        "height": bottom - top - 38
    }

def capture_output_window(window_title="Image Viewer", save_dir="examples/screen/img"):
    hwnd = find_window_by_title(window_title)
    if not hwnd:
        print(f"窗口未找到：{window_title}")
        return
    os.makedirs(save_dir, exist_ok=True)
    region = get_window_rect(hwnd)
    with mss.mss() as sct:
        img = sct.grab(region)
        pil_img = PIL.Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
        timestamp = int(time.time())
        save_path = os.path.join(save_dir, f"frame_{timestamp}.png")
        pil_img.save(save_path)
        print(f"[screenshot] saved to {save_path}")
        return save_path

def receive_full_message(sock):
    """Read full message terminated by newline or until connection closes"""
    buffer = ''
    while True:
        try:
            chunk = sock.recv(1024).decode('utf-8')
            if not chunk:
                break  # connection closed
            buffer += chunk
            if '\n' in buffer:
                break
        except socket.timeout:
            break  # stop waiting
        except Exception as e:
            print(f"Error during receiving: {e}")
            break
    return buffer.strip()

def socket_server():
    """Set up a socket server to listen for keyframe signals"""
    global latest_signal

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('127.0.0.1', 8000))
    server_socket.listen(5)
    print("Socket server started on 127.0.0.1:8000")
    print("Waiting for keyframe signals...")
    
    try:
        while True:
            client_socket, addr = server_socket.accept()
            client_socket.settimeout(1.0)

            try:
                signal = receive_full_message(client_socket)

                if signal:
                    with lock:  # Ensure thread-safe access
                        latest_signal = signal  # Update the latest signal
                    prompt_updated.set()  # Signal that prompt has been updated
                    print(f"Signal received: '{signal}'")
                    client_socket.sendall(b"ACK")
            except Exception as e:
                print(f"Error handling client connection: {str(e)}")
            finally:
                client_socket.close()
    except KeyboardInterrupt:
        print("\nSocket server stopped by user")
    finally:
        server_socket.close()
        print("Socket server closed")

def get_latest_signal():
    """Thread-safe access to the latest signal"""
    with lock:
        return latest_signal
    
# ========================================================

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper

inputs = []

def find_window_by_title(title):
    """查找指定标题的窗口并返回其句柄"""
    result = []
    
    def callback(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if title.lower() in window_title.lower():
                result.append(hwnd)
    
    win32gui.EnumWindows(callback, None)
    return result[0] if result else None

def get_window_rect(hwnd):
    """获取窗口的矩形区域，并去除边框"""
    try:
        # 获取窗口完整矩形
        rect = win32gui.GetWindowRect(hwnd)
        x = rect[0]
        y = rect[1]
        width = rect[2] - x
        height = rect[3] - y
        
        # 调整以去除边框
        adjusted_x = x + 16  # 去除左边框16px
        adjusted_y = y + 50  # 去除顶部30px
        adjusted_width = width - 32  # 去除左右边框各16px
        adjusted_height = height - 66  # 去除顶部30px和底部16px
        
        # 确保宽度和高度不为负
        if adjusted_width <= 0 or adjusted_height <= 0:
            print("=============warning: adjusted window size is invalid, use original size=================")
            return {"top": y, "left": x, "width": width, "height": height}
            
        return {
            "top": adjusted_y, 
            "left": adjusted_x, 
            "width": adjusted_width, 
            "height": adjusted_height
        }
    except Exception as e:
        print(f"=============get window rect failed: {e}=================")
        return None

def screen(
    event: threading.Event,
    height: int = 512,
    width: int = 512,
    window_title: str = "hand detection",
):
    global inputs
    hwnd = find_window_by_title(window_title)
    
    if not hwnd:
        print(f"=============find window '{window_title}' failed=================")
        return
    
    print(f"=============find window '{window_title}'=================")
    
    with mss.mss() as sct:
        while True:
            if event.is_set():
                print("close screen...")
                break
                
            # 实时获取窗口位置，以防窗口移动
            monitor = get_window_rect(hwnd)
            if not monitor:
                time.sleep(0.1)
                continue
                
            img = sct.grab(monitor)
            img = PIL.Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
            img = img.resize((width, height))
            inputs.append(pil2tensor(img))
            
    print('exit: screen')

# Monitoring thread for prompt updates
def prompt_monitor(
    stream,
    base_prompt: str = "lovely hand drawn style, children's illustrations, no one, best quality,masterpiece,ultra high res,delicate eyes,childpaiting,crayon drawing",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution"
):
    """Monitor for prompt updates and take screenshots when needed"""
    screenshot_taken = False
    viewer_ready = False
    
    # Wait for the Image Viewer window to appear
    while not viewer_ready:
        if find_window_by_title("Image Viewer"):
            viewer_ready = True
            print("Image Viewer is ready for screenshot capture")
            time.sleep(2)  # Give it a moment to render first image
        else:
            time.sleep(0.5)
    
    while True:
        if prompt_updated.is_set():
            signal = get_latest_signal()
            if signal:
                new_prompt = f"{base_prompt} {signal}"
                print(f"Updating prompt to: {new_prompt}")
                
                # Update the prompt in the stream
                stream.update_prompt(new_prompt, negative_prompt)
                
                # Wait a moment for the new image to be generated
                time.sleep(3)
                
                # Take a screenshot of the viewer
                screenshot_path = capture_output_window("Image Viewer")
                if screenshot_path:
                    print(f"Screenshot taken after prompt update: {screenshot_path}")
                
                prompt_updated.clear()
        time.sleep(0.5)

def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    close_queue: Queue,
    model_id_or_path: str,
    lora_dict: Optional[Dict[str, float]],
    lora_scale: float,
    prompt: str,
    negative_prompt: str,
    frame_buffer_size: int,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    use_denoising_batch: bool,
    seed: int,
    cfg_type: Literal["none", "full", "self", "initialize"],
    guidance_scale: float,
    delta: float,
    do_add_noise: bool,
    enable_similar_image_filter: bool,
    similar_image_filter_threshold: float,
    similar_image_filter_max_skip_frame: float,
    window_title: str,
) -> None:

    
    global inputs
    # Handle lora_dict and lora_scale properly
    # If lora_dict is a string path, convert it to a dictionary with the scale
    formatted_lora_dict = {lora_dict: lora_scale} if isinstance(lora_dict, str) else lora_dict
    
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=formatted_lora_dict,
        t_index_list=[32, 45],
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=similar_image_filter_threshold,
        similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
    )

    event = threading.Event()
    input_screen = threading.Thread(target=screen, args=(event, height, width, window_title))
    input_screen.start()
    
    # Start the prompt monitor thread
    monitor_thread = threading.Thread(
        target=prompt_monitor, 
        args=(stream, prompt, negative_prompt),
        daemon=True
    )
    monitor_thread.start()
    time.sleep(2)

    while True:
        try:
            if not close_queue.empty(): 
                break
            if len(inputs) < frame_buffer_size:
                time.sleep(0.005)
                continue
            start_time = time.time()
            sampled_inputs = []
            for i in range(frame_buffer_size):
                index = (len(inputs) // frame_buffer_size) * i
                sampled_inputs.append(inputs[len(inputs) - index - 1])
            input_batch = torch.cat(sampled_inputs)
            inputs.clear()
            output_images = stream.stream(
                input_batch.to(device=stream.device, dtype=stream.dtype)
            ).cpu()
            if frame_buffer_size == 1:
                output_images = [output_images]
            for output_image in output_images:
                queue.put(output_image, block=False)

            fps = 1 / (time.time() - start_time)
            fps_queue.put(fps)
        except KeyboardInterrupt:
            break

    print("close image_generation_process...")
    event.set()  #
    input_screen.join()
    print(f"fps: {fps}")

def main(
    model_id_or_path: str = "stabilityai/sd-turbo",
    # lora_dict: str = "D:\work\lora_models_animal_XL.safetensors",
    lora_dict: str = "D:\work\lora_models_SDXL_CrayonPaiting 2.0_2.0.safetensors",
    lora_scale: float = 0.8,
    prompt: str = "lovely hand drawn style, children's illustrations, no one, best quality,masterpiece,ultra high res,delicate eyes,childpaiting,crayon drawing",  # 基础 prompt
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    frame_buffer_size: int = 1,
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    seed: int = 2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    guidance_scale: float = 1.9,
    delta: float = 0.9,
    do_add_noise: bool = False,
    enable_similar_image_filter: bool = True,
    similar_image_filter_threshold: float = 0.99,
    similar_image_filter_max_skip_frame: float = 10,
    window_title: str = "hand detection",
) -> None:

    # Note: We don't check for signals here anymore, as we have a dedicated monitor thread

    ctx = get_context('spawn')
    queue = ctx.Queue()
    fps_queue = ctx.Queue()
    close_queue = Queue()

    process1 = ctx.Process(
        target=image_generation_process,
        args=(
            queue,
            fps_queue,
            close_queue,
            model_id_or_path,
            lora_dict,  # This is a path string that will be converted to a dict
            lora_scale,
            prompt,
            negative_prompt,
            frame_buffer_size,
            width,
            height,
            acceleration,
            use_denoising_batch,
            seed,
            cfg_type,
            guidance_scale,
            delta,
            do_add_noise,
            enable_similar_image_filter,
            similar_image_filter_threshold,
            similar_image_filter_max_skip_frame,
            window_title,
            ),
    )
    process1.start()

    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process2.start()

    # 终止
    process2.join()
    print("process2 已终止。")
    close_queue.put(True)
    print("正在终止 process1...")
    process1.join(5)  # 带超时
    if process1.is_alive():
        print("process1 仍活着。强制终止...")
        process1.terminate()  # 强制终止...
    process1.join()
    print("process1 已终止。")


if __name__ == "__main__":
    # 启动socket服务器线程
    socket_thread = threading.Thread(target=socket_server, daemon=True)
    socket_thread.start()
    
    # 给socket服务器一点时间启动
    time.sleep(1)
    
    # 启动主程序
    fire.Fire(main)