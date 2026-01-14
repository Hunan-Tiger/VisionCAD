from diagnosis.predict import model_load
from medclip.medclip import medclip_load
from mmdet.apis import DetInferencer
from chatcad.cxr.diagnosis import JFinit
import pykinect_azure as pykinect
from restorer.model import Restormer
import torch
import cv2
from Ark.evaluate_mimic import load_ark

import warnings
warnings.filterwarnings("ignore")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def module_load():
    '''
    return: modules(dict)
    '''

    # pykinect.initialize_libraries()
    # device_config = pykinect.default_configuration
    # device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
    # device_config.image_format_color = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    # device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_5
    # interactive_device = [pykinect.start_device(config=device_config), 'kinect']


    interactive_device = None

    if cv2.VideoCapture(0).isOpened():
        interactive_device = [cv2.VideoCapture(0), 'iphone']
        

    # 加载屏幕detect模型
    monitor_detect = DetInferencer(model=r'mmdetection\mmdet\configs\rtmdet\rtmdet_ins_x_8xb16_300e_coco.py',
                                weights=r'mmdetection\rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth')

    
    # load medclip
    medclip, preprocess, tokenizer = medclip_load()

    # 加载修复模型
    restorer = Restormer(inp_channels=3, out_channels=3, dim=48).to(device)
    restorer.load_state_dict(torch.load('restorer/restormer_best.pth', map_location='cpu')['model_state_dict'])
    restorer.eval()

    modules_dict = {'interactive_device': interactive_device,
                  'monitor_detect_module': monitor_detect,
                  'image_restoer_module': restorer,
                  'discrimination_module': [medclip, preprocess, tokenizer],
                }
    return modules_dict


def diagnosis_module_load(modules_dict, PartType):
    # 加载诊断模型
    # 模型对应的疾病需要与Tools中的disease_classes顺序对齐
    if 'Chest X-ray' == PartType:
        # melo模型
        # melo_chest = model_load(device).to(device)
        # melo_chest.swith_lora([0])
        # ark模型
        ark_mimic = load_ark().to(device)

        modules_dict['diagnosis_module'] = {"Chest X-ray": {'ark': [ark_mimic,],}}
        return modules_dict

    elif 'Knee X-ray' == PartType:
        melo_knee = model_load(device)
        melo_knee.swith_lora([1])
        modules_dict['diagnosis_module'] = {"Knee X-ray": {'melo': [melo_knee,],}}
        return modules_dict

