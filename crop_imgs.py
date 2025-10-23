import pykinect_azure as pykinect
import cv2
import os
import pyautogui
import time
import glob


# pykinect.initialize_libraries()
# device_config = pykinect.default_configuration
# device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
# device_config.image_format_color = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
# device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_5
# interactive_device = pykinect.start_device(config=device_config)

# webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
# if not webcam.isOpened():
#     print("can't open the camera!!!")

# # method 1:
# webcam.set(3, 1920)  # width=1920
# webcam.set(4, 1080)  # height=1080
# webcam.set(5, 5)  # frame rate=30

# while True:
#     ret, frame = webcam.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     cv2.imshow('Webcam Feed', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# for n in range(1, 579):
#     save_path = os.path.join('Dataset\OAI_test_by_usb', f"{n:05d}.png")
#     if os.path.exists(save_path):
#         os.remove(save_path)
#     for _ in range(3):
#         ret, frame = webcam.read()

#     ret, frame = webcam.read()
#     cv2.imwrite(save_path, frame)
#     # print(frame.shape[:2])  # just need the first two values.
#     print('-'*70, f" Change to the {n+1}-th image ", '-'*70, '\n')
#     pyautogui.moveTo(3440+1440+200, 720, duration=0.1)  # 3440+1440+200
#     pyautogui.click()
#     pyautogui.press("right")
#     time.sleep(1)

# # Release handle to the webcam
# webcam.release()
# cv2.destroyAllWindows()

# # 处理mp4视频，保存为png格式，按照固定间隔帧数。
# video_path = r"D:\Projects\GlassesCAD\Image\6.5\OAI_Iphone.mp4"  # Change this to your mp4 file path
# output_dir = 'Dataset/OAI_test_by_phone'
# frame_interval = 56  # Save every 30th frame; adjust as needed

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print("Error: Unable to open video file.")
#     exit()

# frame_count = 0
# saved_count = 1

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     if frame_count == 0 or frame_count == 10:
#         save_path = os.path.join(output_dir, f"{saved_count:05d}.png")
#         cv2.imwrite(save_path, frame)
#         print(f"Saved frame {frame_count} as {save_path}")
#         saved_count += 1

#     if (frame_count-10) % frame_interval == 0 and frame_count > 11:
#         save_path = os.path.join(output_dir, f"{saved_count:05d}.png")
#         cv2.imwrite(save_path, frame)
#         print(f"Saved frame {frame_count} as {save_path}")
#         saved_count += 1


#     frame_count += 1

# cap.release()
# cv2.destroyAllWindows()

# # 读取文件夹文件，并从00368.png开始，后面的文件名都需加1，比如00368.png改为00369.png, 00369.png改为00370.png，倒序改，不然会覆盖掉文件
# folder_path = 'Dataset/OAI_test_by_phone'  # Change to your folder path
# start_number = 368

# # Get all png files and sort them in descending order
# png_files = glob.glob(os.path.join(folder_path, '*.png'))
# png_files.sort(reverse=True)

# for file_path in png_files:
#     filename = os.path.basename(file_path)
#     file_number = int(filename.split('.')[0])
    
#     if file_number >= start_number:
#         new_number = file_number + 1
#         new_filename = f"{new_number:05d}.png"
#         new_path = os.path.join(folder_path, new_filename)
        
#         os.rename(file_path, new_path)
#         print(f"Renamed {filename} to {new_filename}")




# # 读取文件夹所有png图片，根据所给的bbox坐标裁剪图片并保存到新文件夹中
# source_folder = 'Dataset/pneu_test_by_usb'  # Source folder with PNG images
# output_folder = 'Dataset/pneu_test_by_usb_cropped'  # Output folder for cropped images

# bbox = (0.509635, 0.521759, 0.414062, 0.723148)  # Normalized coordinates: [x_center, y_center, width, height]

# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# png_files = glob.glob(os.path.join(source_folder, '*.png'))
# png_files.sort()

# for file_path in png_files:
#     filename = os.path.basename(file_path)
#     img = cv2.imread(file_path)

#     if img is not None:
#         height, width = img.shape[:2]
#         # Convert normalized bbox (center format) to pixel coordinates
#         x_center = int(bbox[0] * width)
#         y_center = int(bbox[1] * height)
#         box_width = int(bbox[2] * width)
#         box_height = int(bbox[3] * height)
    
#         x_min = max(0, x_center - box_width // 2)
#         y_min = max(0, y_center - box_height // 2)
#         x_max = min(width, x_center + box_width // 2)
#         y_max = min(height, y_center + box_height // 2)
    
#         print(f"Processing {filename}: ({x_min}, {y_min}), ({x_max}, {y_max})")
#         # Crop the image using the pixel coordinates
#         cropped_img = img[y_min:y_max, x_min:x_max]
    
#         output_path = os.path.join(output_folder, filename)
#         cv2.imwrite(output_path, cropped_img)

#         print(f"Cropped and saved: {filename}")
#     else:
#         print(f"Failed to load: {filename}")


import math
import numpy as np
import cv2
from scipy.spatial import ConvexHull
import pycocotools.mask as maskUtils
from yolov5.detect import run
from mmdet.apis import DetInferencer


monitor_detect = DetInferencer(model=r'D:\Projects\MedAgent\mmdetection\mmdet\configs\rtmdet\rtmdet_ins_x_8xb16_300e_coco.py',
                                weights=r'D:\Projects\MedAgent\mmdetection\rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth')


def localize_monitor(detect=monitor_detect, screenshot_array=None, cls=62):
    results = detect(screenshot_array)
    tv_label_idx = results['predictions'][0]['labels'].index(cls)
    computer_mask = results['predictions'][0]['masks'][tv_label_idx]
    computer_binary_mask = maskUtils.decode(computer_mask)
    coords = np.column_stack(np.where(computer_binary_mask == 1))
    hull = ConvexHull(coords)
    hull_coords = coords[hull.vertices]
    top_left = hull_coords[np.argmin(hull_coords[:, 0] + hull_coords[:, 1])]
    top_right = hull_coords[np.argmin(hull_coords[:, 0] - hull_coords[:, 1])]
    bottom_right = hull_coords[np.argmax(hull_coords[:, 0] + hull_coords[:, 1])]
    bottom_left = hull_coords[np.argmax(hull_coords[:, 0] - hull_coords[:, 1])]
    position = [[top_left[1], top_left[0]], [bottom_left[1], bottom_left[0]], [bottom_right[1], bottom_right[0]], [top_right[1], top_right[0]]]
    screenshot_monitor = perspective_trans(screenshot_array, position)
    return screenshot_monitor


def perspective_trans(img, position):
    def distance(x1, y1, x2, y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

    x1, y1 = position[0][0], position[0][1]
    x2, y2 = position[1][0], position[1][1]
    x3, y3 = position[2][0], position[2][1]
    x4, y4 = position[3][0], position[3][1]

    corners = np.zeros((4, 2), np.float32)
    corners[0] = [x1, y1]
    corners[1] = [x2, y2]
    corners[2] = [x3, y3]
    corners[3] = [x4, y4]

    img_width = distance((x1 + x4) / 2, (y1 + y4) / 2, (x2 + x3) / 2, (y2 + y3) / 2)
    img_height = distance((x1 + x2) / 2, (y1 + y2) / 2, (x4 + x3) / 2, (y4 + y3) / 2)
    # 调整宽高比为16:9
    aspect_ratio = 9 / 16
    if img_width / img_height > aspect_ratio:
        img_width = img_height * aspect_ratio
    else:
        img_height = img_width / aspect_ratio

    corners_trans = np.zeros((4, 2), np.float32)
    corners_trans[0] = [0, 0]
    corners_trans[1] = [img_width - 1, 0]
    corners_trans[2] = [img_width - 1, img_height - 1]
    corners_trans[3] = [0, img_height - 1]

    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
    dst_ = np.transpose(dst, (1, 0, 2))
    return dst_

def localize_image(screenshot_monitor, output_file_path):
    run(
        weights='yolov5/yolo_weights/best.pt',
        source='screenshot/screenshot_monitor.png',
        data='yolov5/data/mydata.yaml',
        imgsz=(640, 640),
        conf_thres=0.70,
        iou_thres=0.45,
        max_det=10,
        device=0,
        view_img=False,
        save_txt=True,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project='screenshot/',
        name='detect',
        exist_ok=True,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
    )
    with open('screenshot/detect/labels/screenshot_monitor.txt', 'r') as file:
        content = file.read()
        yolo_results = content.splitlines()

    img_height, img_width, _ = screenshot_monitor.shape
    output_path = 'screenshot/detect/medical_img'
    os.makedirs(output_path, exist_ok=True)
    medical_img = None
    max_area = -1
    max_bbox = (0, 0, 0, 0)
    for i, line in enumerate(yolo_results):
        yolo_result = line.strip().split()
        class_id = int(yolo_result[0])
        center_x = float(yolo_result[1])
        center_y = float(yolo_result[2])
        width = float(yolo_result[3])
        height = float(yolo_result[4])
        area = width * height
        if i == 0 or area > max_area:
            max_area = area
            max_bbox = (center_x, center_y, width, height)

    x1 = int((max_bbox[0] - max_bbox[2] / 2) * img_width)
    y1 = int((max_bbox[1] - max_bbox[3] / 2) * img_height)
    x2 = int((max_bbox[0] + max_bbox[2] / 2) * img_width)
    y2 = int((max_bbox[1] + max_bbox[3] / 2) * img_height)
    medical_img = screenshot_monitor[y1:y2, x1:x2]
    with open('screenshot/detect/labels/screenshot_monitor.txt', 'w') as file:
        file.truncate(0)
    if max_bbox == (0, 0, 0, 0):
        print("\n-------------------------No medical image detected.----------------------------\n")
        return None, False
    cv2.imwrite(output_file_path, medical_img)
    return medical_img, True



if __name__ == "__main__":
    needed_crop_folder = r"F:\Nodule\kinect\normal_kinect"
    all_captured_images = glob.glob(os.path.join(needed_crop_folder, '*.png'))

    cropped_images_folder = needed_crop_folder + '_cropped'
    if not os.path.exists(cropped_images_folder):
        os.makedirs(cropped_images_folder)
    not_detect_imgs = []
    for image_path in all_captured_images:
        file_name = os.path.basename(image_path)

        output_file_path = os.path.join(cropped_images_folder, file_name)

        screenshot_array = cv2.imread(image_path)
        screenshot_monitor = localize_monitor(screenshot_array=screenshot_array)
        medical_img, flag = localize_image(screenshot_monitor, output_file_path)
        if not flag:
            not_detect_imgs.append(file_name)
            print(f"No medical image detected in {file_name}.")
            continue
        else:
            print(f"Saved cropped image to {output_file_path}")
        
    print(f"----- Total images not detected: {len(not_detect_imgs)}")