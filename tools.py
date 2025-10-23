import os
from PIL import Image
import cv2
from scipy.spatial import ConvexHull
from PIL import Image
from yolov5.detect import run
from pycocotools import mask as maskUtils
import numpy as np
import math
import torch
import tkinter as tk
from tkinter import messagebox
import threading
import subprocess
from chat import init_restore_chat, init_report_generation_chat, add_response_two_image, add_response_restore
from api import inference_chat
import time
from transformers import CLIPTokenizer, CLIPTextModel


import warnings
warnings.filterwarnings("ignore")


from chatcad.cxr.prompt import prob2text
from chatcad.cxr.diagnosis import JFinfer,JFinit
from chatcad.cxr.utils import transform as mimic_cxr_transform

from restorer.restorer import restormer
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from einops import rearrange




API_URL = "https://genaiapi.shanghaitech.edu.cn/api/v1/start"
API_KEY = "Bearer d13074c3b9a34b26813cb9c5648341b9"



device = 'cuda' if torch.cuda.is_available() else 'cpu'
# noME表示疾病不互斥
class Tools:
    '''
    get_monitor_photo, localize_monitor, localize_image, restorer, discriminate, 
    '''
    def __init__(self):
        self.part_types = {'Chest X-ray':0, 'Knee X-ray':1,}
        self.disease_classes = {
            'Chest X-ray':{'pneumonia':{0:'no pneumonia', 1:'pneumonia'},
                           'fivedisease_noME':{0:"Cardiomegaly",1:"Edema",2:"Consolidation",3:"Atelectasis",4:"Pleural Effusion"}},
            'Knee X-ray':{'OAI':{0:'Normal', 1:'Doubtful knee osteoarthritis', 2:'Mild knee osteoarthritis', 3:'Moderate knee osteoarthritis', 4:'Severe knee osteoarthritis'}}
                            }
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def get_monitor_photo(self, output_path, interactive_device):
        if interactive_device is None:
            return 
        save_path = os.path.join(output_path, "screenshot.png")
        if os.path.exists(save_path):
            os.remove(save_path)
        for _ in range(8):
            capture = interactive_device.update()
            ret, _ = capture.get_color_image()
            if not ret:
                continue
        while True:
            capture = interactive_device.update()
            ret, color_image = capture.get_color_image()
            # cv2.imshow('screenshot', color_image)
            # cv2.waitKey(1)
            if not ret:
                continue
            else:
                cv2.imwrite(save_path, color_image)
                break



    def localize_monitor(self, detect, screenshot_array, cls=62):
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
        screenshot_monitor = self.perspective_trans(screenshot_array, position)

        cv2.imwrite("screenshot/screenshot_monitor.png", screenshot_monitor)
        return screenshot_monitor



    def perspective_trans(self, img, position):
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



    def localize_image(self, screenshot_monitor):
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
        return medical_img, True
    
    def text2feature(self, text, model, tokenizer):
        text = tokenizer(text, context_length=256).to(self.device)
        with torch.no_grad():
            output = model.encode_text(text)
        return output

    def restorer(self, rough_image, model, model_cliptextenc, clip_tokenizer):
        input_img = cv2.resize(rough_image, (512, 512))

        restore_chat = add_response_restore('user', init_restore_chat(), Image.fromarray(rough_image))
        text = inference_chat(restore_chat,API_URL, API_KEY)
        #text = ["Modality: X-ray \nRegion: Knee \nView: Frontal \nQuality Issues: \n- Exposure: Overexposure is visible \n- Contrast: Poor contrast between bone and soft tissue \nStructures Affected: Joint space, bone edges \nRequired Improvements: Adjust exposure settings, enhance contrast for better bone and soft tissue distinction"]
        input_prompts = clip_tokenizer(text, return_tensors="pt", padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True).to(device)
        text_input_ids = input_prompts.input_ids
        with torch.no_grad():
            text_embeddings = model_cliptextenc(text_input_ids)[0]

        # Convert to tensor [H,W,C] -> [1,C, H,W]
        input_img = torch.FloatTensor(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)).unsqueeze(0).permute(0,3,1,2) / 255.0
        with torch.no_grad():
            input_img = input_img.to(self.device)
            output = model(input_img, text_embeddings)

            output = output[0]
            out_np = output.cpu().numpy()*255
            out_np = cv2.cvtColor(out_np.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)

            out_np = np.clip(out_np, 0, 255).astype(np.uint8)
        return out_np



    def identification(self, image, model, preprocess, tokenizer, add_labels=None):
        template = 'this is a photo of '
        labels = [str(i) for i in self.part_types.keys()]
        if add_labels:
            labels += add_labels
        context_length = 256
        images = torch.tensor(preprocess(Image.fromarray(image))).unsqueeze(0).to(self.device)
        texts = tokenizer([template + l for l in labels], context_length=context_length).to(self.device)
        with torch.no_grad():
            image_features, text_features, logit_scale = model(images, texts)
            logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
            sorted_indices = torch.argsort(logits, dim=-1, descending=True).squeeze(0)
            logits = logits.cpu().numpy()
            sorted_indices = sorted_indices.cpu().numpy()
        pred = labels[sorted_indices[0]]
        return pred



    def disease_model_preprocess(self, img, transforms_name, img_cfg):
        '''
        不同模型预处理可能不同
        '''

        def general_transform(img, **kwargs):
            img = Resize([224, 224])(img)
            img = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            return img

        def melo(img, **kwargs):
            img = np.array(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)).astype(np.float32)/ 255.0
            img = rearrange(torch.tensor(img, dtype=torch.float32),'h w c ->c h w')
            img = general_transform(img)
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            img = img.to(self.device)
            return img

        def JFinfer(img, img_cfg,**kwargs):
            img = mimic_cxr_transform(img, img_cfg)
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            img = img.to(self.device)
            return img

        transforms_dict = {
            'general': general_transform,
            'melo': melo,
            'JFinfer': JFinfer
        }

        return transforms_dict[transforms_name](img, img_cfg=img_cfg) #preprocess_img


    def diagnosis(self, image, models_dict, PartType,):
        '''
        image: cv2, BGR,
        models: 对应模态的不同疾病分类器, dict
        PartType: str
        '''
        classes_result = ''
        disease_list = [str(key) for key in self.disease_classes[PartType].keys()]
        with torch.no_grad():
            idx = 0
            for name, models in models_dict.items():
                img_cfg = None if len(models) == 1 else models[1]
                pre_img = self.disease_model_preprocess(image, name, img_cfg)
                model = models[0].eval()
                logits = model(pre_img)

                if 'noME' in disease_list[idx]:
                    # 当疾病分类独立时
                    prob = list(torch.sigmoid(torch.Tensor(logits)).cpu().numpy())
                    pred = [int(_) for _ in prob]
                else:
                    prob = list(torch.softmax(logits[0], dim=0).cpu().numpy())
                    pred = int(torch.argmax(logits[0], dim=0).cpu().numpy())

                classes_result += self.prob2text(prob, PartType, disease_list[idx])
                idx += 1
                
        return classes_result
    

    def prob2text(self, prob, PartType, disease_name):
        classes_result = ''
        for i in range(len(prob)):
            if prob[i] < 0.2:
                classes_result += f"Hardly {self.disease_classes[PartType][disease_name][i]}. "
            elif prob[i]>=0.2 and prob[i] <0.5:
                classes_result += f"Small possibility of {self.disease_classes[PartType][disease_name][i]}. "
            elif prob[i]>=0.5 and prob[i] <0.9:
                classes_result += f"It is likely to have {self.disease_classes[PartType][disease_name][i]}. "
            else:
                classes_result += f"Definitely have {self.disease_classes[PartType][disease_name][i]}. "
        classes_result += '\n'
        return classes_result
    


    def report_generation(self, medical_image, diagnosis_hint, model="gpt-4o"):
        ref_report = "Patient is status post median sternotomy and CABG. Mild cardiomegaly is similar. The aorta remains tortuous, and the mediastinal and hilar contours are unchanged. Pulmonary vasculature is not engorged. There is minimal atelectasis at the lung bases without focal consolidation. No pleural effusion or pneumothorax is detected. Degenerative changes are seen throughout the thoracic spine."
        ref_medical_img = Image.open(r"D:\Projects\chatcad\ref\02f3df2f-ff2bc640-5f173dca-eaff305d-73b20ae1.jpg").convert('L')
        ref_prompt = 'The diagnostic report for this medical image is as follows:\n'+ref_report

        report_chat = init_report_generation_chat()
        prompt = 'Please use a style similar to the above report to diagnose this medical image. (Do not say anything similar: The diagnostic report for this medical image is as follows)'
        chat_diagnosis = add_response_two_image("user", ref_prompt, prompt, diagnosis_hint, report_chat, ref_medical_img, Image.fromarray(medical_image).convert('L'))
        report = inference_chat(chat_diagnosis, API_URL, API_KEY, model)
        return report



    def multimedia_dispaly(self, report_generation):
        stop_event = threading.Event()
        speecher_thread = threading.Thread(target=self.multimedia_speaker, args=(report_generation, stop_event))
        speecher_thread.start()
        msg_box = tk.Tk()
        msg_box.withdraw()
        msg_box.attributes('-topmost', True)
        messagebox.showinfo("Report", report_generation)
        msg_box.destroy()
        stop_event.set()
        speecher_thread.join()


    def multimedia_speaker(self, TEXT, stop_event):
        cmd = f'edge-tts --text \"{TEXT}\" --write-media result.wav'
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=60)
            if result.returncode == 0:
                play_cmd = 'start /min mpv.exe result.wav'
                play_process = subprocess.Popen(play_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                while not stop_event.is_set():
                    continue
                try:
                    subprocess.run('taskkill /f /im mpv.exe', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=10)
                except subprocess.TimeoutExpired:
                    print("Taskkill command timed out")
                except Exception as e:
                    print(f"An error occurred while killing mpv.exe: {e}")
            else:
                print(f"Error executing command: {result.stderr.decode()}")
        except subprocess.TimeoutExpired:
            print("Command timed out")
        except Exception as e:
            print(f"An error occurred: {e}")
        if os.path.exists("result.wav"):
            os.remove("result.wav")
    

    def diff_two_frames(self, frame1, frame2):
        # 将图像调整为相同大小，如果它们的大小不同
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        # 将两个图像帧转换为灰度图像
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # 使用高斯模糊来减少噪声
        gray_frame1 = cv2.GaussianBlur(gray_frame1, (5, 5), 0)
        gray_frame2 = cv2.GaussianBlur(gray_frame2, (5, 5), 0)
        # 计算两个灰度图像帧之间的绝对差异
        diff = cv2.absdiff(gray_frame1, gray_frame2)
        # 对差异图像进行二值化处理
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        # 使用形态学操作去除小的噪点
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # 计算二值化图像中非零像素的数量
        non_zero_count = cv2.countNonZero(thresh)
        # 计算图像的总像素数
        total_pixels = gray_frame1.shape[0] * gray_frame1.shape[1]
        # 计算差异像素的百分比
        diff_percentage = (non_zero_count / total_pixels) * 100
        # 返回布尔值，指示是否存在显著差异, 以及差异百分比
        return diff_percentage < 5, diff_percentage



class Agent():
    def __init__(self):
        self.system_prompt = {
            'role': 'You are a medical system agent, assisting doctors in completing the process of diagnosing medical images.',
            'environment': 'You are in a hospital, and doctor\'s head mounted camera will upload you a photo, in which the subject is a computer monitor displaying a medical image.',
            'tools': 'You have a set of tools to help you complete the diagnosis process, \
                including ##Locator## (Locate the medical image on the display and crop it out),\
                ##Restorer## (Restoration of captured medical images),\
                ##Discrimination Module## (Identify the type and location of medical images, such as Chest X-ray),\
                ##Diagnostic Module## (Diagnosis of disease classification based on medical images), \
                ##Report Generation Module## (Generate a diagnostic report based on the classification diagnosis results combined with the medical images themselves), and \
                ##Multimedia## (Display diagnostic reports through text pop ups or speaker playback, etc) for presenting reports.',
            'task': 'You need to use the current tools ##step by step## to diagnose the medical images in the captured photo.',
        }