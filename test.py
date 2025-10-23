import os
import random
import json
import time
import pyautogui
import cv2
import numpy as np
import pykinect_azure as pykinect
from PIL import Image
from img_localization import localize_img, localize_computer_transformer
from api import inference_chat
from controller import kinect_screenshot
from prompt import get_diagnosis_prompt, get_identify_prompt
from chat import init_diagnosis_chat, add_response, add_response_two_image
from mmdet.apis import DetInferencer
from diagnosis.predict import diagnosis, model_load
from medclip.medclip import medclip_load, medclip_identify
from chatcad.cxr.diagnosis import JFinfer,JFinit
from tqdm import tqdm


API_URL = "https://genaiapi.shanghaitech.edu.cn/api/v1/start"
API_KEY = "Bearer d6df978ccba44a44a7e079e156f326e3"

melo = model_load(device="cuda")
inferencer = DetInferencer(model=r'D:\Projects\MedAgent\mmdetection\mmdet\configs\rtmdet\rtmdet_ins_x_8xb16_300e_coco.py',
                            weights=r'D:\Projects\MedAgent\mmdetection\rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth')
screenshot = "./screenshot"

def collect_data(files):
    pykinect.initialize_libraries()
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
    device_config.image_format_color = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_5

    iter = 0
    current_files = files
    base_path = 'Image\\Test'
    save_imgs_path = os.path.join(base_path, current_files)
    os.makedirs(save_imgs_path, exist_ok=True)
    files = len(os.listdir(os.path.join('Image', current_files)))

    while iter<files:
        if iter < 903:
            iter += 1
            continue
        device = pykinect.start_device(config=device_config)
        kinect_screenshot(screenshot, device)
        screenshot_array = cv2.imread("screenshot/screenshot.png") # return ndarray
        screenshot_computer = localize_computer_transformer(inferencer, screenshot_array, 62) # return ndarray
        medical_imgs = localize_img(screenshot_computer) # return list[ndarray]
        file_name = os.listdir(os.path.join('Image', current_files))[iter].split('.')[0] + '.png'
        medical_img = cv2.cvtColor(medical_imgs[0], cv2.COLOR_BGR2RGB)
        cv2.imshow('medical_img', medical_img)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(save_imgs_path, f'{file_name}'), medical_img)
        print(f"Save {file_name} successfully.\n")
        
        pyautogui.FAILSAFE = False
        pyautogui.moveTo(3440+1440+200, 700)
        pyautogui.click()
        pyautogui.press("right")
        device.close()
        iter += 1
        if iter >= 1000:
            break

def melo_test(type_, melo, files):
    base_path = f'Image\\test\\{files}'
    imgs = os.listdir(base_path)
    length = len(imgs)
    acc = 0
    for img in imgs:
        img_path = os.path.join(base_path, img)
        label = int(img.split('_')[-1].split('.')[0])
        img = cv2.imread(img_path)
        _, pred, medical_image = diagnosis(img, melo, type_)
        break
        if label == pred:
            acc += 1
    return acc, length

def medclip_test(type_, files, model, preprocess, tokenizer):
    base_path = f'Image\\test\\{files}'
    imgs = os.listdir(base_path)
    length = len(imgs)
    acc = 0
    label = type_ 
    for img in imgs:
        img_path = os.path.join(base_path, img)
        img = cv2.imread(img_path)
        pred = medclip_identify(img, model, preprocess, tokenizer)
        if label == pred:
            acc += 1
    return acc, length


if __name__ == '__main__':
    medclip, preprocess, tokenizer = medclip_load()
    img_model, imgcfg=JFinit(r'D:\Projects\MedAgent\chatcad\cxr\config\JF.json',r'D:\Projects\MedAgent\chatcad\weights\JFchexpert.pth')
    img_model = img_model.to("cuda")


    ref_report = "Patient is status post median sternotomy and CABG.  Mild cardiomegaly is\n similar.  The aorta remains tortuous, and the mediastinal and hilar contours\n are unchanged.  Pulmonary vasculature is not engorged.  There is minimal\n atelectasis at the lung bases without focal consolidation.  No pleural\n effusion or pneumothorax is detected.  Degenerative changes are seen\n throughout the thoracic spine."
    ref_medical_img = Image.open(r"D:\Projects\chatcad\ref\02f3df2f-ff2bc640-5f173dca-eaff305d-73b20ae1.jpg").convert('L')
    ref_prompt = 'The diagnostic report for this medical image is as follows:\n'+ref_report
    chat_diagnosis = init_diagnosis_chat()
    prompt = 'Please follow the style of the report above to diagnose this medical image. (Do not say anything similar: The diagnostic report for this medical image is as follows)'

    dcm_list = json.load(open(r'Image\MIMIC\dcm_list.json'))
    annotation = json.load(open(r'Image\MIMIC\annotation.json'))

    imgs_path = r'Image\MIMIC\ori_imgs'
    imgs = os.listdir(imgs_path)
    iter = 0
    for idx in tqdm(range(len(imgs))):
        iter += 1
        img = imgs[idx]
        img_path = os.path.join(imgs_path, img)
        medical_image = cv2.imread(img_path)
        print(medical_image.shape)
        # # 原始图resize
        # medical_image = cv2.resize(medical_image, (512, 512))

        id_ = dcm_list[idx]
        gt_report, gt = next((item["report"], item["gt"]) for item in annotation if item["img"] == id_)

        identify_result = 'Chest X-ray'#medclip_identify(medical_image, medclip, preprocess, tokenizer)
        diagnosis_hint, _, medical_image = diagnosis(image=medical_image, melo=melo, cxr_model=img_model, cxr_cfg=imgcfg, ImageType=identify_result, device="cuda")


        # chat_diagnosis = init_diagnosis_chat()
        # chat_diagnosis = add_response_two_image("user", ref_prompt, prompt, diagnosis_hint, chat_diagnosis, ref_medical_img, Image.fromarray(medical_image, mode='L'))
        # report = inference_chat(chat_diagnosis, API_URL, API_KEY, model="gpt-4o")

        # result = {
        #         "id": id_,
        #         "observation": diagnosis_hint,
        #         "generated_report": report,
        #         "ground_report": gt_report,
        #         "gt": gt
        #     }
        
        # with open(f'Image\MIMIC\gene_report_GT.json', 'a', encoding='utf-8') as f:
        #     json.dump(result, f, indent=4, ensure_ascii=False)
        #     f.write(',')
        #     f.write('\n')


    # melo:
    # Pneu: 556/624=0.891 original: 0.923
    # Derma: 767/1000=0.767 original: 0.888
    # OAI: 629/1000=0.629 original: 0.651
    # OCT: 778/1000=0.778 original: 0.896
    # Path: 894/1000=0.894 original: 0.961
    # Retina: 241/400=0.602 original 0.658


    # medclip:
    # Pneu_Accuracy: 624/624
    # Derma_Accuracy: 848/1000
    # OAI_Accuracy: 980/1000
    # OCT_Accuracy: 998/1000
    # Path_Accuracy: 935/1000
    # Retina_Accuracy: 400/400
