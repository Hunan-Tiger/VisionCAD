from diagnosis.predict import model_load
from medclip.medclip import medclip_load
from mmdet.apis import DetInferencer
from chatcad.cxr.diagnosis import JFinit
import pykinect_azure as pykinect
from restorer.Text2Restore import Restormer
import torch
from transformers import CLIPTokenizer, CLIPTextModel


import warnings
warnings.filterwarnings("ignore")

def module_load():
    '''
    return: modules(dict)
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载摄像头
    try:
        pykinect.initialize_libraries()
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
        device_config.image_format_color = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_5
        interactive_device = pykinect.start_device(config=device_config)

    except Exception as e:
        print(f"Error initializing Kinect: {e}")
        interactive_device = None

    # 加载屏幕detect模型
    monitor_detect = DetInferencer(model=r'D:\Projects\MedAgent\mmdetection\mmdet\configs\rtmdet\rtmdet_ins_x_8xb16_300e_coco.py',
                                weights=r'D:\Projects\MedAgent\mmdetection\rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth')
    
    # load medclip
    medclip, preprocess, tokenizer = medclip_load()

    # load CLIP
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(device)


    # 加载修复模型
    restorer = Restormer(inp_channels=3, out_channels=3, dim=48).to(device)
    restorer.load_state_dict(torch.load('restorer/ours_best.pth', map_location='cpu')['model_state_dict'])
    restorer.eval()

    modules_dict = {'interactive_device': interactive_device,
                  'monitor_detect_module': monitor_detect,
                  'clip_module': [clip_text_encoder, clip_tokenizer],
                  'image_restoer_module': restorer,
                  'discrimination_module': [medclip, preprocess, tokenizer],
                }
    return modules_dict

def diagnosis_module_load(modules_dict, PartType):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 加载诊断模型
    # 模型对应的疾病需要与Tools中的disease_classes顺序对齐
    if 'Chest X-ray' == PartType:
        melo_chest = model_load(device)
        melo_chest.swith_lora([0])
        mimic_model, mimic_cfg=JFinit(r'D:\Projects\MedAgent\chatcad\cxr\config\JF.json',r'D:\Projects\MedAgent\chatcad\weights\JFchexpert.pth')
        mimic_model = mimic_model.to(device)

        modules_dict['diagnosis_module'] = {"Chest X-ray": {'melo': [melo_chest,], 'JFinfer': [mimic_model, mimic_cfg],}}
        return modules_dict

    elif 'Knee X-ray' == PartType:
        melo_knee = model_load(device)
        melo_knee.swith_lora([1])
        modules_dict['diagnosis_module'] = {"Knee X-ray": {'melo': [melo_knee,],}}
        return modules_dict

