import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.spatial import ConvexHull
from PIL import Image
import os
from yolov5.detect import run
from pycocotools import mask as maskUtils

def prepocess_computer(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute histogram to find the most common pixel value (mode)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    mode_pixel = np.argmax(hist)

    # Shift the most common pixel value to 0 (black)
    shift_value = mode_pixel
    corrected_image = cv2.addWeighted(gray_image, 1, gray_image, 0, -shift_value)

    # Apply contrast stretching to enhance the appearance
    min_pixel, max_pixel = np.min(corrected_image), np.max(corrected_image)
    stretched_image = ((corrected_image - min_pixel) / (max_pixel - min_pixel) * 255).astype(np.uint8)
    return stretched_image


def localize_computer_transformer(detect, screenshot_array, cls=62):# 62 is the label index of TV
    results = detect(screenshot_array)

    tv_label_idx = results['predictions'][0]['labels'].index(cls)
    computer_mask = results['predictions'][0]['masks'][tv_label_idx]
    computer_binary_mask = maskUtils.decode(computer_mask)
    # 找到 mask_ 中值为 1 的所有像素点的坐标
    coords = np.column_stack(np.where(computer_binary_mask == 1))
    # 计算凸包
    hull = ConvexHull(coords)
    # 获取凸包顶点的坐标
    hull_coords = coords[hull.vertices]
    # 找到左上角、左下角、右下角、右上角坐标
    top_left = hull_coords[np.argmin(hull_coords[:, 0] + hull_coords[:, 1])]
    top_right = hull_coords[np.argmin(hull_coords[:, 0] - hull_coords[:, 1])]
    bottom_right = hull_coords[np.argmax(hull_coords[:, 0] + hull_coords[:, 1])]
    bottom_left = hull_coords[np.argmax(hull_coords[:, 0] - hull_coords[:, 1])]
    position = [[top_left[1], top_left[0]],[bottom_left[1], bottom_left[0]],[bottom_right[1], bottom_right[0]],[top_right[1], top_right[0]]]
    screenshot_computer = perspective_trans(screenshot_array, position) # np.array

    cv2.imwrite("screenshot/computer_screenshot.png", screenshot_computer)
    return screenshot_computer


def enhance_image(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Split the LAB image to different channels
    l, a, b = cv2.split(lab)
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # Merge the CLAHE enhanced L-channel with a and b channels
    limg = cv2.merge((cl, a, b))
    # Convert image from LAB color space back to RGB color space
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Apply sharpening filter
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_image, -1, kernel)
    
    return sharpened


def localize_img(screenshot_computer, idx):
    # png = cv2.imread('screenshot/computer_screenshot.png')
    # cv2.imwrite('screenshot/computer_screenshot.png', enhance_image(png))
    run(
    weights='yolo_weights/best.pt', #5m
    source='screenshot/computer_screenshot.png',
    data = 'data/mydata.yaml',
    imgsz=(640, 640),
    conf_thres=0.70,  # 置信度阈值  
    iou_thres=0.45,  # IoU 阈值
    max_det=10,  # 最大检测数量
    device=0,  # 设备
    view_img=False,  # 是否显示图像
    save_txt=True,  # 是否保存检测结果为文本
    save_conf=False,  # 是否保存置信度
    save_crop=False,  # 是否保存裁剪后的检测结果
    nosave=False,  # 是否不保存图像
    classes=None,  # 检测的类别
    agnostic_nms=False,  # 类别无关的 NMS
    augment=False,  # 是否使用数据增强
    visualize=False,  # 是否可视化特征图
    update=False,  # 是否更新模型
    project='screenshot/',  # 项目目录
    name='detect',  # 结果保存目录
    exist_ok=True,  # 是否覆盖已存在的目录
    line_thickness=3,  # 线条厚度
    hide_labels=False,  # 是否隐藏标签
    hide_conf=False,  # 是否隐藏置信度
    half=False,  # 是否使用半精度 FP16
    dnn=False,  # 是否使用 OpenCV DNN 模式
)
    # 读取 YOLO 的检测结果
    with open('screenshot/detect/labels/computer_screenshot.txt', 'r') as file:
        content = file.read()
        yolo_results = content.splitlines()

    # 读取原图像
    img_height, img_width, _ = screenshot_computer.shape
    output_path = 'screenshot\detect\medical_img'
    os.makedirs(output_path, exist_ok=True)
    medical_img = None
    # 逐行处理 YOLO 的检测结果
    max_area = -1
    max_bbox = (0,0,0,0)
    for i, line in enumerate(yolo_results):
        yolo_result = line.strip().split()

        # 解析 YOLO 的检测结果
        class_id = int(yolo_result[0])
        center_x = float(yolo_result[1])
        center_y = float(yolo_result[2])
        width = float(yolo_result[3])
        height = float(yolo_result[4])
        # 计算边界框的面积
        area = width * height
        if i == 0 or area > max_area:
            max_area = area
            max_bbox = (center_x, center_y, width, height)

    # 计算边界框的实际像素坐标
    x1 = int((max_bbox[0] - max_bbox[2] / 2) * img_width)
    y1 = int((max_bbox[1] - max_bbox[3] / 2) * img_height)
    x2 = int((max_bbox[0] + max_bbox[2] / 2) * img_width)
    y2 = int((max_bbox[1] + max_bbox[3] / 2) * img_height)
    # 裁剪图像
    medical_img = screenshot_computer[y1:y2, x1:x2]
    # 清空txt文件内容
    with open('screenshot/detect/labels/computer_screenshot.txt', 'w') as file:
        file.truncate(0)
    if medical_img is None:
        print("\n-------------------------No medical image detected.----------------------------\n")
        medical_img = screenshot_computer
        return medical_img, False
    return medical_img, True


def perspective_trans(img, position):
    def distance(x1,y1,x2,y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))    

    x1, y1 = position[0][0], position[0][1]
    x2, y2 = position[1][0], position[1][1]
    x3, y3 = position[2][0], position[2][1]
    x4, y4 = position[3][0], position[3][1]

    corners = np.zeros((4,2), np.float32)
    corners[0] = [x1, y1]
    corners[1] = [x2, y2]
    corners[2] = [x3, y3]
    corners[3] = [x4, y4]

    img_width = distance((x1+x4)/2, (y1+y4)/2, (x2+x3)/2, (y2+y3)/2)
    img_height = distance((x1+x2)/2, (y1+y2)/2, (x4+x3)/2, (y4+y3)/2)

    corners_trans = np.zeros((4,2), np.float32)
    # 保证和position中的点的顺序一致
    corners_trans[0] = [0, 0]
    corners_trans[1] = [img_width - 1, 0]
    corners_trans[2] = [img_width - 1, img_height - 1]
    corners_trans[3] = [0, img_height - 1]

    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
    dst_ = np.transpose(dst, (1, 0, 2))
    return dst_


# 逆时针排序四个点，输出左上角、左下角、右下角、右上角
def order_point(coor):
    arr = np.array(coor).reshape((4, 2))
    centroid = np.mean(arr, axis=0)
    # 计算每个点相对于质心的角度
    theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    # 按角度逆时针排序
    sort_indices = np.argsort(theta)
    sort_points = arr[sort_indices]
    # 找到左上角点（x 和 y 之和最小的点）
    start_index = np.argmin(np.sum(sort_points, axis=1))
    # 调整顺序，从左上角开始
    sort_points = np.roll(sort_points, -start_index, axis=0)
    return sort_points.astype('float32')

def ocr(image_pil, ocr_detection, ocr_recognition):
    text_data = []
    coordinate = []
    image_full = np.array(image_pil)
    image_full = cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB)
    det_result = ocr_detection(image_full)
    det_result = det_result['polygons'] 
    for i in range(det_result.shape[0]):
        pts = order_point(det_result[i]) # 得到四个点的坐标（逆时针）
        image_crop = crop_image(image_full, pts) # 切割出文本区域（透视变换）
        try:
            result = ocr_recognition(image_crop)['text'][0] # 识别文本
        except:
            continue

        box = [int(e) for e in list(pts.reshape(-1))]
        box = [box[0], box[1], box[4], box[5]] # 提取左上角和右下角坐标
        
        text_data.append(result)
        coordinate.append(box)
        
    return text_data, coordinate

def filename_extractor(text):
    for t in text:
        if 'png' in t or 'jpg' in t:
           return t.split('.')[0]


def crop_image(img, position):
    def distance(x1,y1,x2,y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))    
    position = position.tolist()
    for i in range(4):
        for j in range(i+1, 4):
            if(position[i][0] > position[j][0]):
                tmp = position[j]
                position[j] = position[i]
                position[i] = tmp
    if position[0][1] > position[1][1]:
        tmp = position[0]
        position[0] = position[1]
        position[1] = tmp

    if position[2][1] > position[3][1]:
        tmp = position[2]
        position[2] = position[3]
        position[3] = tmp

    x1, y1 = position[0][0], position[0][1]
    x2, y2 = position[2][0], position[2][1]
    x3, y3 = position[3][0], position[3][1]
    x4, y4 = position[1][0], position[1][1]

    corners = np.zeros((4,2), np.float32)
    corners[0] = [x1, y1]
    corners[1] = [x2, y2]
    corners[2] = [x4, y4]
    corners[3] = [x3, y3]

    img_width = distance((x1+x4)/2, (y1+y4)/2, (x2+x3)/2, (y2+y3)/2)
    img_height = distance((x1+x2)/2, (y1+y2)/2, (x4+x3)/2, (y4+y3)/2)

    corners_trans = np.zeros((4,2), np.float32)
    corners_trans[0] = [0, 0]
    corners_trans[1] = [img_width - 1, 0]
    corners_trans[2] = [0, img_height - 1]
    corners_trans[3] = [img_width - 1, img_height - 1]

    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
    return dst


# def show_mask(mask, image, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     plt.gca().imshow(mask_image)
#     plt.show()
 
# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels == 1]
#     neg_points = coords[labels == 0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
#                linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
#                linewidth=1.25)