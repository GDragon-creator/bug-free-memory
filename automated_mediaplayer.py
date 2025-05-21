import cv2
import mediapipe as mp
import pyautogui
import time
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from pynput import keyboard, mouse
import json
import os
from PIL import Image, ImageTk
import locale
import configparser
import logging
import sys

#配置全局日志
logging.basicConfig(
    filename='app.log',  # 日志文件路径
    level=logging.DEBUG,  # 日志级别（DEBUG/INFO/WARNING/ERROR）
    format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# --- 加载语言文件 ---
try:
    with open("languages.json", "r", encoding="utf-8") as f:
        translations = json.load(f)
except FileNotFoundError:
    print("错误: 未找到 languages.json 文件。请确保文件存在于脚本同级目录下。")
    logger.error("错误: 未找到 languages.json 文件。请确保文件存在于脚本同级目录下。")
    exit(1)
except json.JSONDecodeError:
    print("错误: languages.json 文件格式不正确。请检查文件内容是否为有效的 JSON 格式。")
    logger.error("错误: languages.json 文件格式不正确。请检查文件内容是否为有效的 JSON 格式。")
    exit(1)


def count_fingers(lst, handedness_label="Unknown"):  # 接收手的左右标签
    cnt = 0
    if not lst or not lst.landmark:
        return 0

    try:
        # 垂直阈值：基于手腕到中指指根的Y轴距离的比例，判断手指是否向上伸直
        # landmark[0] 是 WRIST, landmark[9] 是 MIDDLE_FINGER_MCP
        # TIP.y < MCP.y (因为Y轴向下为正，所以 TIP.y 较小表示手指向上)
        # (MCP.y - TIP.y) > vertical_thresh
        vertical_thresh = abs(lst.landmark[0].y - lst.landmark[9].y) / 2.8  # 可调整分母

        # 食指 (MCP:5, TIP:8)
        if (lst.landmark[5].y - lst.landmark[8].y) > vertical_thresh:
            cnt += 1
        # 中指 (MCP:9, TIP:12)
        if (lst.landmark[9].y - lst.landmark[12].y) > vertical_thresh:
            cnt += 1
        # 无名指 (MCP:13, TIP:16)
        if (lst.landmark[13].y - lst.landmark[16].y) > vertical_thresh:
            cnt += 1
        # 小指 (MCP:17, TIP:20)
        if (lst.landmark[17].y - lst.landmark[20].y) > vertical_thresh:
            cnt += 1

        # 拇指逻辑 (基于手的左右和X轴坐标)
        # landmark[2] 是 THUMB_MCP (拇指掌骨关节)
        # landmark[4] 是 THUMB_TIP (拇指指尖)
        # landmark[5] 是 INDEX_FINGER_MCP
        # landmark[17] 是 PINKY_MCP

        # 水平阈值参考：基于食指指根到小指指根的X轴距离（近似手掌宽度）的比例
        # 调整这个比例因子可以改变拇指判断的灵敏度
        thumb_horizontal_ref_dist = abs(lst.landmark[5].x - lst.landmark[17].x)
        thumb_thresh = thumb_horizontal_ref_dist * 0.3  # 例如，拇指伸出超过手掌宽度的30%

        # 图像已经水平翻转 (frm = cv2.flip(frm, 1))
        # MediaPipe报告的 "Left" 指的是物理上的左手, "Right" 指的是物理上的右手

        if handedness_label == "Left":  # 物理左手 (在翻转的屏幕上显示为右手)
            # 拇指伸出时，指尖(4)的x坐标会大于其MCP(2)的x坐标 (在屏幕上向右)
            if (lst.landmark[4].x - lst.landmark[2].x) > thumb_thresh:
                cnt += 1
        elif handedness_label == "Right":  # 物理右手 (在翻转的屏幕上显示为左手)
            # 拇指伸出时，指尖(4)的x坐标会小于其MCP(2)的x坐标 (在屏幕上向左)
            if (lst.landmark[2].x - lst.landmark[4].x) > thumb_thresh:
                cnt += 1
        # 如果handedness_label是"Unknown", 则不特意判断拇指或使用一个通用但不一定准确的规则
        # 为简单起见，这里未知手型时不计数拇指，依赖上面四个手指的计数

    except IndexError:
        logger.warning("手指关键点索引错误", exc_info=True)
        print("手指关键点索引错误。")  # Landmarks可能不完整
        return 0
    except Exception as e:
        logger.error(f"手指计数错误: {str(e)}", exc_info=True)
        print(f"手指计数错误: {e}")
        return 0
    return cnt


# 颜色配置
primary_color = "#3B82F6"
secondary_color = "#64748B"
accent_color = "#F97316"
success_color = "green"
error_color = "#EF4444"
neutral_bg = "#F8FAFC"
card_bg = "#FFFFFF"

# 保存绑定状态
capturing_for_finger = None
finger_actions = {i: None for i in range(1, 6)}
action_labels = {}
set_buttons = {}
status_label = None
gui_root = None
current_keys = set()
scroll_action = None
config_done_and_start = False
listener_keyboard = None
listener_mouse = None
title_label = None
control_header = None
start_button = None
menubar = None
options_menu = None
about_menu = None
language_menu = None
about_window = None
contact_window = None
gesture_labels = {}

# 当前语言，默认为简体中文
current_language = "zh-CN"


# --- 保存设置到文件 ---
def save_settings():
    global finger_actions
    try:
        settings = {str(i): finger_actions[i] for i in range(1, 6)}
        with open("settings.json", "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
        print(translations[current_language]["export_success"].format("settings.json"))
    except Exception as e:
        print(translations[current_language]["export_failed"].format(e))
        messagebox.showerror(translations[current_language]["error_title"],
                             translations[current_language]["export_failed"].format(e))


# --- 从文件加载设置 ---
def load_settings():
    global finger_actions, current_language
    config = configparser.ConfigParser()
    # 尝试加载 config.ini 文件
    if os.path.exists("config.ini"):
        config.read("config.ini", encoding="utf-8")
        if "Settings" in config and "language" in config["Settings"]:
            lang = config["Settings"]["language"]
            language_map = {
                "auto": "auto",
                "zh-CN": "zh-CN",
                "zh-TW": "zh-TW",
                "en": "en"
            }
            if lang in language_map:
                current_language = lang
            else:
                current_language = "zh-CN"  # 默认语言
    else:
        current_language = "zh-CN"  # 默认语言，如果文件不存在

    # 加载手势设置
    try:
        if os.path.exists("settings.json"):
            with open("settings.json", "r", encoding="utf-8") as f:
                settings = json.load(f)
            for i in range(1, 6):
                key = str(i)
                if key in settings:
                    action = settings[key]
                    if action is None or (
                            isinstance(action, dict) and
                            "type" in action and
                            action["type"] in ["key", "combo", "mouse_scroll", "mouse_click"] and
                            (("value" in action and isinstance(action["value"], str)) or
                             ("button" in action and isinstance(action["button"], str)))
                    ):
                        finger_actions[i] = action
                    else:
                        print(f"无效的设置项，手指 {i}: {action}")
                        finger_actions[i] = None
            print(translations[current_language]["import_success"].format("settings.json"))
        else:
            print("未找到 settings.json，使用默认设置")
        logger.info("成功加载配置文件 settings.json")
    except Exception as e:
        logger.error(f"加载配置失败: {str(e)}", exc_info=True)  # exc_info=True 记录完整堆栈
        finger_actions = {i: None for i in range(1, 6)}


# --- 重置设置 ---
def reset_settings():
    global finger_actions, action_labels, set_buttons
    finger_actions = {i: None for i in range(1, 6)}
    for i in range(1, 6):
        if i in action_labels and action_labels[i].winfo_exists():
            action_labels[i].config(text=translations[current_language]["not_set"], foreground=error_color)
        if i in gesture_labels and gesture_labels[i].winfo_exists():
            gesture_labels[i].config(text=translations[current_language]["finger_gesture"].format(i))
        if i in set_buttons and set_buttons[i].winfo_exists():
            set_buttons[i].config(text=translations[current_language]["set_action"],
                                  command=lambda i_copy=i: set_action_mode(i_copy))
    if status_label and status_label.winfo_exists():
        status_label.config(text=translations[current_language]["status_reset"])
    save_settings()


def format_keys(keys):
    special_keys_order = ['ctrl', 'alt', 'shift', 'tab', 'enter']
    special_keys = []
    regular_keys = []
    for key_val in keys:
        if key_val in special_keys_order:
            special_keys.append(key_val)
        else:
            regular_keys.append(key_val)
    sorted_special_keys = sorted(special_keys,
                                 key=lambda x: special_keys_order.index(x) if x in special_keys_order else 999)
    sorted_regular_keys = sorted(regular_keys)
    result = sorted_special_keys + sorted_regular_keys
    return '+'.join(result)


def on_key_press(key_event):
    global capturing_for_finger
    if capturing_for_finger is None:
        return
    try:
        if hasattr(key_event, 'vk') and 65 <= key_event.vk <= 90:
            key_name = chr(key_event.vk).lower()
        elif hasattr(key_event, 'vk') and (48 <= key_event.vk <= 57 or 96 <= key_event.vk <= 105):
            if 48 <= key_event.vk <= 57:
                key_name = chr(key_event.vk)
            else:
                key_name = chr(key_event.vk - 48)
        elif hasattr(key_event, 'char') and key_event.char:
            key_name = key_event.char.lower()
        elif hasattr(key_event, 'name'):
            key_name = key_event.name.lower()
        else:
            key_name = str(key_event).replace("'", "")
        if 'ctrl' in key_name:
            key_name = 'ctrl'
        elif 'shift' in key_name:
            key_name = 'shift'
        elif 'alt' in key_name:
            key_name = 'alt'
        if key_name in ['cmd', 'meta', 'super']:
            stop_capture_mode()
            if gui_root and gui_root.winfo_exists():
                messagebox.showwarning(translations[current_language]["invalid_key"],
                                       translations[current_language]["invalid_key_message"])
            print(f"检测到无效按键: {key_name}")
            logger.error(f"检测到无效按键: {key_name}")
            return
        current_keys.add(key_name)
    except Exception as e:
        print(f"键处理错误: {e}")
        logger.error(f"键处理错误: {e}")


def on_key_release(key_event):
    global capturing_for_finger
    if capturing_for_finger is None:
        return
    if any(k in ['cmd', 'meta', 'super'] for k in current_keys):
        print("组合键包含 Win 键，取消绑定。")
        stop_capture_mode()
        return
    if not current_keys:
        print("未捕获到有效按键组合。")
        logger.error("未捕获到有效按键组合。")
        return
    action_type = 'key' if len(current_keys) == 1 else 'combo'
    action = {'type': action_type, 'value': format_keys(current_keys)}
    update_action_display_and_clear_conflicts(capturing_for_finger, action)
    stop_capture_mode()


def on_scroll(x, y, dx, dy):
    global capturing_for_finger
    if capturing_for_finger is None:
        return
    direction = 'scroll_up' if dy > 0 else 'scroll_down'
    action = {'type': 'mouse_scroll', 'value': direction}
    update_action_display_and_clear_conflicts(capturing_for_finger, action)
    stop_capture_mode()


def on_click(x, y, button, pressed):
    global capturing_for_finger, gui_root, set_buttons
    if capturing_for_finger is None or not pressed:
        return
    if gui_root:
        for i in range(1, 6):
            btn = set_buttons[i]
            if btn and btn.winfo_exists():
                btn_x = btn.winfo_rootx()
                btn_y = btn.winfo_rooty()
                btn_width = btn.winfo_width()
                btn_height = btn.winfo_height()
                if btn_x <= x <= btn_x + btn_width and btn_y <= y <= btn_y + btn_height:
                    return
    action_val = {'type': 'mouse_click', 'button': str(button)}
    update_action_display_and_clear_conflicts(capturing_for_finger, action_val)
    stop_capture_mode()


def update_action_display_and_clear_conflicts(finger_num, new_action):
    global finger_actions, action_labels, set_buttons
    for f_idx, existing_action in finger_actions.items():
        if f_idx != finger_num and existing_action == new_action:
            finger_actions[f_idx] = None
            if f_idx in action_labels and action_labels[f_idx].winfo_exists():
                action_labels[f_idx].config(text=translations[current_language]["not_set"], foreground=error_color)
            if f_idx in set_buttons and set_buttons[f_idx].winfo_exists():
                set_buttons[f_idx].config(text=translations[current_language]["set_action"],
                                          command=lambda i=f_idx: set_action_mode(i))
    finger_actions[finger_num] = new_action
    desc = ''
    if new_action['type'] == 'key':
        desc = translations[current_language]["key"].format(new_action['value'])
    elif new_action['type'] == 'combo':
        desc = translations[current_language]["combo"].format(new_action['value'])
    elif new_action['type'] == 'mouse_scroll':
        desc = translations[current_language]["mouse_scroll_up"] if new_action['value'] == 'scroll_up' else \
            translations[current_language]["mouse_scroll_down"]
    elif new_action['type'] == 'mouse_click':
        btn_name = new_action['button'].replace('Button.', '')
        desc = translations[current_language]["mouse_click_left"] if btn_name == 'left' else \
            translations[current_language]["mouse_click_right"] if btn_name == 'right' else \
                translations[current_language]["mouse_click_middle"]
    if finger_num in action_labels and action_labels[finger_num].winfo_exists():
        action_labels[finger_num].config(text=desc, foreground=success_color)
    if finger_num in set_buttons and finger_num in set_buttons and set_buttons[finger_num].winfo_exists():
        set_buttons[finger_num].config(text=translations[current_language]["cancel_action"],
                                       command=lambda i=finger_num: cancel_action(i))


def cancel_action(finger_num):
    global finger_actions, action_labels, set_buttons, status_label
    finger_actions[finger_num] = None
    if finger_num in action_labels and action_labels[finger_num].winfo_exists():
        action_labels[finger_num].config(text=translations[current_language]["not_set"], foreground=error_color)
    if finger_num in set_buttons and set_buttons[finger_num].winfo_exists():
        set_buttons[finger_num].config(text=translations[current_language]["set_action"],
                                       command=lambda i=finger_num: set_action_mode(i))
    if status_label and status_label.winfo_exists():
        status_label.config(text=translations[current_language]["status_cancel"].format(finger_num))


def stop_capture_mode():
    global capturing_for_finger, listener_keyboard, listener_mouse, status_label, set_buttons, finger_actions
    current_finger = capturing_for_finger
    capturing_for_finger = None
    current_keys.clear()
    try:
        if listener_keyboard:
            listener_keyboard.stop()
            listener_keyboard = None
        if listener_mouse:
            listener_mouse.stop()
            listener_mouse = None
    except Exception as e:
        print(f"停止监听器时出错: {e}")
        logger.error(f"停止监听器时出错: {e}")
    if gui_root and gui_root.winfo_exists():
        if current_finger and current_finger in action_labels and action_labels[current_finger].winfo_exists():
            if finger_actions.get(current_finger) is None:
                action_labels[current_finger].config(text=translations[current_language]["not_set"], foreground=error_color)
        if status_label and status_label.winfo_exists():
            status_label.config(text=translations[current_language]["status_default"])
        for i in range(1, 6):
            if i in set_buttons and set_buttons[i].winfo_exists():
                set_buttons[i].config(state=tk.NORMAL)
                if finger_actions.get(i) is not None:
                    set_buttons[i].config(text=translations[current_language]["cancel_action"],
                                          command=lambda i_copy=i: cancel_action(i_copy))
                else:
                    set_buttons[i].config(text=translations[current_language]["set_action"],
                                          command=lambda i_copy=i: set_action_mode(i_copy))


def set_action_mode(finger_num):
    global capturing_for_finger, listener_keyboard, listener_mouse, action_labels, status_label, set_buttons
    capturing_for_finger = finger_num
    current_keys.clear()
    if finger_num in action_labels and action_labels[finger_num].winfo_exists():
        action_labels[finger_num].config(text=translations[current_language]["status_capturing"], foreground="blue")
    if status_label and status_label.winfo_exists():
        status_label.config(text=translations[current_language]["status_setting"].format(finger_num))
    for i in range(1, 6):
        if i in set_buttons and set_buttons[i].winfo_exists():
            set_buttons[i].config(state=tk.DISABLED)
    listener_keyboard = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    listener_mouse = mouse.Listener(on_scroll=on_scroll, on_click=on_click)
    listener_keyboard.start()
    listener_mouse.start()


def start_camera_detection():
    global config_done_and_start, gui_root
    all_set = True
    for i in range(1, 6):
        if finger_actions.get(i) is None:
            all_set = False
            break
    if not all_set:
        if not messagebox.askyesno(translations[current_language]["confirm_start"],
                                   translations[current_language]["confirm_start_message"]):
            return
    config_done_and_start = True
    save_settings()
    if gui_root:
        gui_root.quit()
        gui_root.destroy()
        gui_root = None


# --- 转存配置 ---
def export_settings():
    global finger_actions
    file_path = tk.filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
    if file_path:
        try:
            settings = {str(i): finger_actions[i] for i in range(1, 6)}
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            print(translations[current_language]["export_success"].format(file_path))
        except Exception as e:
            messagebox.showerror(translations[current_language]["error_title"],
                                 translations[current_language]["export_failed"].format(e))
            logger.error("error_title export_failed")


# --- 加载配置 ---
def import_settings():
    global finger_actions, action_labels
    file_path = tk.filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if file_path:
        original_finger_actions = finger_actions.copy()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            valid = True
            expected_keys = {"1", "2", "3", "4", "5"}
            if not isinstance(settings, dict) or set(settings.keys()) != expected_keys:
                valid = False
            else:
                for key in expected_keys:
                    value = settings.get(key)
                    if value is None:
                        continue
                    if not isinstance(value, dict) or "type" not in value:
                        valid = False
                        break
                    action_type = value["type"]
                    if action_type not in ["key", "combo", "mouse_scroll", "mouse_click"]:
                        valid = False
                        break
                    if action_type in ["key", "combo"]:
                        if "value" not in value or not isinstance(value["value"], str):
                            valid = False
                            break
                    elif action_type == "mouse_scroll":
                        if "value" not in value or value["value"] not in ["scroll_up", "scroll_down"]:
                            valid = False
                            break
                    elif action_type == "mouse_click":
                        if "button" not in value or not isinstance(value["button"], str):
                            valid = False
                            break
            if not valid:
                raise ValueError(translations[current_language]["import_format_error"])
            finger_actions = {i: settings.get(str(i), None) for i in range(1, 6)}
            for i in range(1, 6):
                action = finger_actions.get(i)
                desc = translations[current_language]["not_set"]
                foreground = error_color
                if action:
                    if action['type'] == 'key':
                        desc = translations[current_language]["key"].format(action['value'])
                    elif action['type'] == 'combo':
                        desc = translations[current_language]["combo"].format(action['value'])
                    elif action['type'] == 'mouse_scroll':
                        desc = translations[current_language]["mouse_scroll_up"] if action['value'] == 'scroll_up' else \
                            translations[current_language]["mouse_scroll_down"]
                    elif action['type'] == 'mouse_click':
                        btn_name = action['button'].replace('Button.', '')
                        desc = translations[current_language]["mouse_click_left"] if btn_name == 'left' else \
                            translations[current_language]["mouse_click_right"] if btn_name == 'right' else \
                                translations[current_language]["mouse_click_middle"]
                    foreground = success_color
                if i in action_labels and action_labels[i].winfo_exists():
                    action_labels[i].config(text=desc, foreground=foreground)
            print(translations[current_language]["import_success"].format(file_path))
        except Exception as e:
            finger_actions = original_finger_actions
            messagebox.showerror(translations[current_language]["error_title"],
                                 translations[current_language]["import_failed"].format(e))
            logger.error("error_title import_failed")


# --- 切换语言 ---
def switch_language(lang):
    global current_language
    language_map = {
        "自动": "auto",
        "简体中文": "zh-CN",
        "繁体中文": "zh-TW",
        "English": "en"
    }
    if lang == "自动":
        loc = locale.getdefaultlocale()[0]
        if loc and loc.startswith("zh"):
            if loc == "zh_TW":
                current_language = "zh-TW"
            else:
                current_language = "zh-CN"
        else:
            current_language = "en"
    else:
        current_language = language_map[lang]

    # 保存语言到 config.ini
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    if not config.has_section("Settings"):
        config.add_section("Settings")
    config.set("Settings", "language", current_language)
    with open("config.ini", "w", encoding="utf-8") as configfile:
        config.write(configfile)

    # 更新界面文本
    if title_label and title_label.winfo_exists():
        title_label.config(text=translations[current_language]["title"])
    if control_header and control_header.winfo_exists():
        control_header.config(text=translations[current_language]["control_header"])
    if status_label and status_label.winfo_exists():
        status_label.config(text=translations[current_language]["status_default"])
    if start_button and start_button.winfo_exists():
        start_button.config(text=translations[current_language]["start_button"])
    for i in range(1, 6):
        # 更新“指手势”标签
        if i in gesture_labels and gesture_labels[i].winfo_exists():
            gesture_labels[i].config(text=translations[current_language]["finger_gesture"].format(i))
        # 更新动作标签
        if i in action_labels and action_labels[i].winfo_exists():
            action = finger_actions.get(i)
            desc = translations[current_language]["not_set"]
            foreground = error_color
            if action:
                if action['type'] == 'key':
                    desc = translations[current_language]["key"].format(action['value'])
                elif action['type'] == 'combo':
                    desc = translations[current_language]["combo"].format(action['value'])
                elif action['type'] == 'mouse_scroll':
                    desc = translations[current_language]["mouse_scroll_up"] if action['value'] == 'scroll_up' else \
                        translations[current_language]["mouse_scroll_down"]
                elif action['type'] == 'mouse_click':
                    btn_name = action['button'].replace('Button.', '')
                    desc = translations[current_language]["mouse_click_left"] if btn_name == 'left' else \
                        translations[current_language]["mouse_click_right"] if btn_name == 'right' else \
                            translations[current_language]["mouse_click_middle"]
                foreground = success_color
            action_labels[i].config(text=desc, foreground=foreground)
        if i in set_buttons and set_buttons[i].winfo_exists():
            if finger_actions.get(i) is not None:
                set_buttons[i].config(text=translations[current_language]["cancel_action"])
            else:
                set_buttons[i].config(text=translations[current_language]["set_action"])

        # 更新菜单
        menubar.entryconfig(1, label=translations[current_language]["options_menu"])
        menubar.entryconfig(2, label=translations[current_language]["language_menu"])
        menubar.entryconfig(3, label=translations[current_language]["about_menu"])
        options_menu.entryconfig(0, label=translations[current_language]["save_config"])
        options_menu.entryconfig(1, label=translations[current_language]["export_config"])
        options_menu.entryconfig(2, label=translations[current_language]["import_config"])
        options_menu.entryconfig(3, label=translations[current_language]["reset_config"])
        options_menu.entryconfig(4, label=translations[current_language]["exit"])
        about_menu.entryconfig(0, label=translations[current_language]["about_intro"])
        about_menu.entryconfig(1, label=translations[current_language]["about_contact"])


# --- 显示“介绍”界面 ---
def show_about_intro():
    global about_window
    if about_window and about_window.winfo_exists():
        about_window.lift()  # 如果窗口已存在，置顶
        return

    about_window = tk.Toplevel(gui_root)
    about_window.title(translations[current_language]["about_intro"])
    about_window.geometry("400x300")
    about_window.resizable(False, False)

    # 介绍内容
    intro_text = translations[current_language].get("about_intro_text",
                                                    "This is a gesture-controlled media player application developed using Python.\n"
                                                    "It uses MediaPipe for hand tracking and pyautogui for action execution.\n"
                                                    "Version: 1.0 | Date: May 20, 2025")
    label = ttk.Label(about_window, text=intro_text, justify="left", wraplength=380, padding=10)
    label.pack(expand=True)

    # 添加关闭按钮
    close_button = ttk.Button(about_window, text=translations[current_language]["close"],
                              command=about_window.destroy, style="Primary.TButton")
    close_button.pack(pady=10)


# --- 显示“联系我们”界面 ---
def show_contact_us():
    global contact_window
    if contact_window and contact_window.winfo_exists():
        contact_window.lift()  # 如果窗口已存在，置顶
        return

    contact_window = tk.Toplevel(gui_root)
    contact_window.title(translations[current_language]["about_contact"])
    contact_window.geometry("350x450")
    contact_window.resizable(False, False)

    # 加载微信图片（请替换为实际路径）
    wechat_image_path = "./images/wechat.png"  # 占位符，需替换为你的微信二维码图片路径
    try:
        image = Image.open(wechat_image_path)
        image = image.resize((250, 350), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        image_label = ttk.Label(contact_window, image=photo)
        image_label.image = photo  # 保持引用，避免垃圾回收
        image_label.pack(pady=10)
    except FileNotFoundError:
        error_label = ttk.Label(contact_window, text=translations[current_language].get("image_not_found",
                                                                                        "WeChat QR code image not found. Please place 'wechat_qr.png' in the 'images' folder."),
                                foreground=error_color)
        error_label.pack(pady=10)
        logger.error("image_not_found")

    # 添加关闭按钮
    close_button = ttk.Button(contact_window, text=translations[current_language]["close"],
                              command=contact_window.destroy, style="Primary.TButton")
    close_button.pack(pady=10)


def setup_gui():
    global gui_root, status_label, action_labels, set_buttons, title_label, control_header, start_button, menubar, options_menu, about_menu, language_menu
    # 初始化 config.ini 文件
    config = configparser.ConfigParser()
    if not os.path.exists("config.ini"):
        config.add_section("Settings")
        config.set("Settings", "language", "zh-CN")  # 默认语言
        with open("config.ini", "w", encoding="utf-8") as configfile:
            config.write(configfile)
    load_settings()
    gui_root = tk.Tk()
    gui_root.title(translations[current_language]["title"])
    # 添加菜单栏
    menubar = tk.Menu(gui_root)
    gui_root.config(menu=menubar)
    options_menu = tk.Menu(menubar, tearoff=0)
    language_menu = tk.Menu(menubar, tearoff=0)
    about_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label=translations[current_language]["options_menu"], menu=options_menu)
    menubar.add_cascade(label=translations[current_language]["language_menu"], menu=language_menu)
    menubar.add_cascade(label=translations[current_language]["about_menu"], menu=about_menu)
    options_menu.add_command(label=translations[current_language]["save_config"], command=save_settings)
    options_menu.add_command(label=translations[current_language]["export_config"], command=export_settings)
    options_menu.add_command(label=translations[current_language]["import_config"], command=import_settings)
    options_menu.add_command(label=translations[current_language]["reset_config"], command=reset_settings)
    options_menu.add_command(label=translations[current_language]["exit"], command=on_gui_close)
    language_menu.add_command(label=translations[current_language]["auto_option"] if "auto_option" in translations[
        current_language] else "自动",
                              command=lambda: switch_language("自动"))
    language_menu.add_command(label=translations[current_language]["zh_cn_option"] if "zh_cn_option" in translations[
        current_language] else "简体中文",
                              command=lambda: switch_language("简体中文"))
    language_menu.add_command(label=translations[current_language]["zh_tw_option"] if "zh_tw_option" in translations[
        current_language] else "繁体中文",
                              command=lambda: switch_language("繁体中文"))
    language_menu.add_command(label=translations[current_language]["en_option"] if "en_option" in translations[
        current_language] else "English",
                              command=lambda: switch_language("English"))
    about_menu.add_command(label=translations[current_language]["about_intro"], command=show_about_intro)
    about_menu.add_command(label=translations[current_language]["about_contact"], command=show_contact_us)
    # 设置窗口大小和背景颜色
    gui_root.geometry("700x600")
    gui_root.configure(bg=neutral_bg)
    gui_root.resizable(True, True)
    # 确保中文显示正常
    font_families = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Inter", "Arial"]
    default_font = ("SimHei", 10)
    style = ttk.Style()
    style.theme_use('clam')
    style.configure(".", font=default_font, background=neutral_bg)
    style.configure("Title.TLabel",
                    font=(font_families[0], 20, "bold"),
                    foreground="#1E293B",
                    background=neutral_bg)
    style.configure("Card.TFrame", background=card_bg)
    style.configure("CardHeader.TLabel",
                    font=(font_families[0], 14, "bold"),
                    foreground="#1E293B",
                    background=card_bg)
    style.configure("Primary.TButton",
                    font=(font_families[0], 11, "bold"),
                    foreground="white",
                    background=primary_color,
                    borderwidth=0,
                    focuscolor="none")
    style.configure("Secondary.TButton",
                    font=(font_families[0], 10),
                    foreground="white",
                    background=secondary_color,
                    borderwidth=0,
                    focuscolor="none")
    style.configure("Accent.TButton",
                    font=(font_families[0], 11, "bold"),
                    foreground="white",
                    background=accent_color,
                    borderwidth=0,
                    focuscolor="none")
    style.map("Primary.TButton",
              background=[("active", "#2563EB"), ("disabled", "#94A3B8")])
    style.map("Secondary.TButton",
              background=[("active", "#475569"), ("disabled", "#94A3B8")])
    style.map("Accent.TButton",
              background=[("active", "#EA580C"), ("disabled", "#94A3B8")])
    style.configure("Status.TLabel",
                    font=(font_families[0], 11),
                    foreground=secondary_color,
                    background=neutral_bg)
    style.configure("Error.TLabel",
                    font=(font_families[0], 11),
                    foreground=error_color,
                    background=neutral_bg)
    style.configure("Success.TLabel",
                    font=(font_families[0], 11),
                    foreground=success_color,
                    background=neutral_bg)
    style.configure("TCombobox",
                    fieldbackground=card_bg,
                    background=card_bg,
                    bordercolor="#E2E8F0",
                    darkcolor="#E2E8F0",
                    lightcolor="#E2E8F0")
    style.configure("BlueButton.TButton",
                    font=(font_families[0], 10),
                    foreground="white",
                    background=primary_color,
                    borderwidth=0,
                    focuscolor="none")
    style.map("BlueButton.TButton",
              background=[("active", "#60A5FA"), ("disabled", "#E2E8F0")])
    # 创建主框架
    main_frame = ttk.Frame(gui_root, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    # 顶部标题
    title_frame = ttk.Frame(main_frame, padding="0 0 0 5")
    title_frame.pack(fill=tk.X, pady=(0, 20))
    title_label = ttk.Label(title_frame, text=translations[current_language]["title"], style="Title.TLabel")
    title_label.pack(side=tk.LEFT)
    # 创建内容区域
    content_frame = ttk.Frame(main_frame)
    content_frame.pack(fill=tk.BOTH, expand=True)
    # 左侧控制面板
    control_frame = ttk.Frame(content_frame, style="Card.TFrame", padding="15")
    control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
    control_header = ttk.Label(control_frame, text=translations[current_language]["control_header"],
                               style="CardHeader.TLabel")
    control_header.pack(anchor=tk.W, pady=(0, 10))
    ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 15))
    for i in range(1, 6):
        gesture_frame = ttk.Frame(control_frame, padding="0 5")
        gesture_frame.pack(fill=tk.X, pady=(0, 10))
        label = ttk.Label(gesture_frame, text=translations[current_language]["finger_gesture"].format(i), width=16,
                          wraplength=150)
        label.pack(side=tk.LEFT, padx=(0, 10))
        gesture_labels[i] = label  # 保存引用
        action = finger_actions.get(i)
        desc = translations[current_language]["not_set"]
        foreground = error_color
        if action:
            if action['type'] == 'key':
                desc = translations[current_language]["key"].format(action['value'])
            elif action['type'] == 'combo':
                desc = translations[current_language]["combo"].format(action['value'])
            elif action['type'] == 'mouse_scroll':
                desc = translations[current_language]["mouse_scroll_up"] if action['value'] == 'scroll_up' else \
                    translations[current_language]["mouse_scroll_down"]
            elif action['type'] == 'mouse_click':
                btn_name = action['button'].replace('Button.', '')
                desc = translations[current_language]["mouse_click_left"] if btn_name == 'left' else \
                    translations[current_language]["mouse_click_right"] if btn_name == 'right' else \
                        translations[current_language]["mouse_click_middle"]
            foreground = success_color
        label = ttk.Label(gesture_frame, text=desc, foreground=foreground, width=40, anchor="w",
                          relief="solid", borderwidth=1, padding="5")
        label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        action_labels[i] = label
        btn = ttk.Button(gesture_frame, text=translations[current_language]["set_action"], style="BlueButton.TButton",
                         command=lambda i_copy=i: set_action_mode(i_copy))
        btn.pack(side=tk.RIGHT)
        set_buttons[i] = btn
    status_frame = ttk.Frame(control_frame, padding="5")
    status_frame.pack(fill=tk.X, pady=(15, 0))
    status_label = ttk.Label(status_frame, text=translations[current_language]["status_default"],
                             style="Status.TLabel")
    status_label.pack(anchor=tk.W)
    button_frame = ttk.Frame(control_frame, padding="5")
    button_frame.pack(fill=tk.X, pady=(15, 0))
    start_button = ttk.Button(
        button_frame,
        text=translations[current_language]["start_button"],
        style="Accent.TButton",
        command=start_camera_detection,
        padding=(10, 12)
    )
    start_button.pack(fill=tk.X)
    gui_root.protocol("WM_DELETE_WINDOW", on_gui_close)
    gui_root.mainloop()


def on_gui_close():
    global config_done_and_start, gui_root
    try:
        if messagebox.askokcancel(translations[current_language]["exit_confirm"],
                              translations[current_language]["exit_confirm_message"]):
            save_settings()
            logger.info("用户主动关闭GUI界面")
            config_done_and_start = False
            if gui_root:
                gui_root.quit()
                gui_root.destroy()
                gui_root = None
    except Exception as e:
        logger.error(f"关闭GUI时出错: {str(e)}", exc_info=True)


def draw_chinese_text(img_cv, text, pos, font_size=30, color=(255, 0, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("./fonts/simhei.ttf", font_size)
    except IOError:
        logger.error("IO Exception")
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
            if text != translations[current_language]["font_fallback"]:
                print(translations[current_language]["font_fallback"])
                logger.error("font_fallback")
        except IOError:
            print(translations[current_language]["font_missing"])
            logger.error("font_missing")
            cv2.putText(img_cv, translations[current_language]["font_error"], (pos[0], pos[1] + font_size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size / 30, color, 2)
            return img_cv
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    try:
        setup_gui()
        if not config_done_and_start:
            print(translations[current_language]["config_canceled"])
            exit()

        print(translations[current_language]["config_done"])
        print("已配置的操作:", finger_actions)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(translations[current_language]["camera_error"])
            exit()

        drawing = mp.solutions.drawing_utils
        hands_module = mp.solutions.hands
        hand_obj = hands_module.Hands(max_num_hands=2, min_detection_confidence=0.7,
                                      min_tracking_confidence=0.6)  # 使用 hand_gesture_reader.py 的更高置信度

        start_init = False
        prev_cnt = -1
        last_action_time = 0
        action_hold_time = 0.25
        min_action_interval = 0.4
        action_gesture_start_time = 0
        two_hands_detected_start_time = 0.0
        exit_countdown_duration = 3.0

        cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)

        while True:
            current_time = time.time()
            ret, frm = cap.read()
            if not ret:
                print(translations[current_language]["frame_error"])
                break

            frm = cv2.flip(frm, 1)
            rgb_frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            res = hand_obj.process(rgb_frm)

            finger_count_display = translations[current_language]["fingers_na"]
            hands_label_display = translations[current_language]["no_hands"]
            exit_countdown_text = ""
            current_handedness_label = "Unknown"  # 用于传递给 count_fingers

            num_hands_detected = 0
            if res.multi_hand_landmarks:
                num_hands_detected = len(res.multi_hand_landmarks)

                for i, hand_landmarks in enumerate(res.multi_hand_landmarks):
                    drawing.draw_landmarks(frm, hand_landmarks, hands_module.HAND_CONNECTIONS)

                if num_hands_detected == 2:
                    hands_label_display = translations[current_language]["both_hands"]
                    if two_hands_detected_start_time == 0.0:
                        two_hands_detected_start_time = current_time
                    elapsed_two_hands_time = current_time - two_hands_detected_start_time
                    remaining_time_for_exit = exit_countdown_duration - elapsed_two_hands_time
                    if remaining_time_for_exit > 0:
                        exit_countdown_text = translations[current_language]["exit_countdown"].format(
                            remaining_time_for_exit)
                    else:
                        exit_countdown_text = translations[current_language]["exiting"]
                    if elapsed_two_hands_time >= exit_countdown_duration:
                        print("检测到双手持续3秒，程序退出。")
                        break
                    finger_count_display = translations[current_language]["fingers_na_both"]
                    start_init = False
                    prev_cnt = -1
                    action_gesture_start_time = 0
                elif num_hands_detected == 1:
                    two_hands_detected_start_time = 0.0
                    hand_keyPoints = res.multi_hand_landmarks[0]

                    if res.multi_handedness and len(res.multi_handedness) > 0:
                        handedness_info = res.multi_handedness[0].classification[0]
                        current_handedness_label = handedness_info.label  # "Left" or "Right"
                        if current_handedness_label == "Left":
                            hands_label_display = translations[current_language]["left_hand"]
                        elif current_handedness_label == "Right":
                            hands_label_display = translations[current_language]["right_hand"]
                        else:
                            hands_label_display = current_handedness_label
                    else:
                        hands_label_display = translations[current_language]["unknown_hand"]

                    cnt = count_fingers(hand_keyPoints, current_handedness_label)
                    finger_count_display = translations[current_language]["fingers_count"].format(cnt)

                    if prev_cnt != cnt:
                        action_gesture_start_time = current_time
                        prev_cnt = cnt
                        start_init = True

                    if start_init and (current_time - action_gesture_start_time) > action_hold_time:
                        action_to_perform = finger_actions.get(cnt)
                        if action_to_perform:
                            if (current_time - last_action_time) > min_action_interval:
                                print(translations[current_language]["execute_action"].format(cnt, hands_label_display,
                                                                                              action_to_perform))
                                logger.error("execute_action")
                                try:
                                    if action_to_perform['type'] == 'key':
                                        pyautogui.press(action_to_perform['value'])
                                    elif action_to_perform['type'] == 'combo':
                                        keys_to_press = action_to_perform['value'].split('+')
                                        for k_val in keys_to_press:
                                            pyautogui.keyDown(k_val)
                                        # time.sleep(0.05)  # 小延迟以确保组合键生效
                                        for k_val in reversed(keys_to_press):
                                            pyautogui.keyUp(k_val)
                                    elif action_to_perform['type'] == 'mouse_scroll':
                                        if action_to_perform['value'] == 'scroll_up':
                                            pyautogui.scroll(120)  # 使用 hand_gesture_reader.py 的标准滚动单位
                                        elif action_to_perform['value'] == 'scroll_down':
                                            pyautogui.scroll(-120)
                                    elif action_to_perform['type'] == 'mouse_click':
                                        button_val = action_to_perform['button'].replace('Button.', '')
                                        pyautogui.click(button=button_val)
                                    last_action_time = current_time
                                except Exception as e:
                                    print(f"使用 pyautogui 执行操作时出错: {e}")
                                    logger.error(f"使用 pyautogui 执行操作时出错: {e}")
                        start_init = False
            else:
                two_hands_detected_start_time = 0.0
                hands_label_display = translations[current_language]["no_hands"]
                if prev_cnt != 0:
                    prev_cnt = 0
                    start_init = False
                    action_gesture_start_time = 0
                finger_count_display = translations[current_language]["fingers_count"].format(0)

            frm = draw_chinese_text(frm, finger_count_display, pos=(10, 30), font_size=28, color=(255, 0, 0))
            frm = draw_chinese_text(frm, hands_label_display, pos=(10, 70), font_size=28, color=(0, 255, 0))
            if exit_countdown_text:
                frm = draw_chinese_text(frm, exit_countdown_text, pos=(10, 110), font_size=28, color=(0, 0, 255))

            try:
                if cv2.getWindowProperty("Gesture Control", cv2.WND_PROP_VISIBLE) <= 0:
                    print(translations[current_language]["window_closed"])
                    break
            except cv2.error:
                print(translations[current_language]["window_destroyed"])
                break

            cv2.imshow("Gesture Control", frm)

            key_input = cv2.waitKey(1) & 0xFF
            if key_input == 27:
                break
            elif key_input == ord('r') or key_input == ord('R'):
                print(translations[current_language]["reconfig"])
                if cap.isOpened(): cap.release()
                cv2.destroyAllWindows()

                finger_actions = {i: None for i in range(1, 6)}
                action_labels.clear()
                set_buttons.clear()
                gesture_labels.clear()  # 清空 gesture_labels
                config_done_and_start = False
                two_hands_detected_start_time = 0.0

                if listener_keyboard or listener_mouse:
                    stop_capture_mode()

                setup_gui()

                if not config_done_and_start:
                    print(translations[current_language]["config_canceled"])
                    exit()

                print(translations[current_language]["reconfig_done"])
                print("已配置的操作:", finger_actions)
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print(translations[current_language]["camera_error"])
                    exit()
                hand_obj = hands_module.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
                cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)

                start_init = False
                prev_cnt = -1
                last_action_time = 0
                action_gesture_start_time = 0

        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

        if listener_keyboard or listener_mouse:
            stop_capture_mode()
        print(translations[current_language]["app_end"])
    except Exception as e:
        logger.critical(f"全局未处理异常: {str(e)}", exc_info=True)
        messagebox.showerror("系统错误", "程序发生严重错误，已记录日志到 app.log")
        sys.exit(1)
