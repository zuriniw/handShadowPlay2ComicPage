import os
import sys
import threading
import time
import tkinter as tk
from multiprocessing import Queue
from typing import List
from PIL import Image, ImageTk
from streamdiffusion.image_utils import postprocess_image
import mss
import win32gui

sys.path.append(os.path.join(os.path.dirname(__file__),  ".."))


def update_image(image_data: Image.Image, label: tk.Label) -> None:
    width = 512
    height = 512
    tk_image = ImageTk.PhotoImage(image_data, size=width)
    label.configure(image=tk_image, width=width, height=height)
    label.image = tk_image  # keep a reference


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


def capture_output_window(window_title="Image Viewer", save_dir='examples/screen/img'):
    hwnd = find_window_by_title(window_title)
    if not hwnd:
        print(f"窗口未找到：{window_title}")
        return
    os.makedirs(save_dir, exist_ok=True)
    region = get_window_rect(hwnd)
    with mss.mss() as sct:
        img = sct.grab(region)
        pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
        existing = [f for f in os.listdir(save_dir) if f.endswith(".png") and f.startswith("frame_")]
        indices = [int(f.split("_")[1].split(".")[0]) for f in existing if f.split("_")[1].split(".")[0].isdigit()]
        next_index = max(indices) + 1 if indices else 0
        save_path = os.path.join(save_dir, f"frame_{next_index}.png")
        pil_img.save(save_path)
        print(f"[screenshot] saved to {save_path}")
        return save_path


def _receive_images(
    queue: Queue, fps_queue: Queue, label: tk.Label, fps_label: tk.Label
) -> None:
    while True:
        try:
            if not queue.empty():
                label.after(
                    0,
                    update_image,
                    postprocess_image(queue.get(block=False), output_type="pil")[0],
                    label,
                )
            if not fps_queue.empty():
                fps_label.config(text=f"FPS: {fps_queue.get(block=False):.2f}")

            time.sleep(0.0005)
        except KeyboardInterrupt:
            return


def receive_images(queue: Queue, fps_queue: Queue) -> None:
    root = tk.Tk()
    root.title("Image Viewer")
    label = tk.Label(root)
    fps_label = tk.Label(root, text="FPS: 0")
    label.grid(column=0)
    fps_label.grid(column=1)

    def on_closing():
        print("window closed")
        root.quit()
        return

    thread = threading.Thread(
        target=_receive_images, args=(queue, fps_queue, label, fps_label), daemon=True
    )
    thread.start()

    try:
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    except KeyboardInterrupt:
        return
