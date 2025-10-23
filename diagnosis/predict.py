from diagnosis.lora import LoRA_ViT_timm_x
import timm
import torch
import numpy as np
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from einops import rearrange
from controller import correct_gray, correct_color
from PIL import Image
from chatcad.cxr.utils import transform
import cv2
from restorer.restorer import restormer

melo_paths = [
    'diagnosis/weights/Pneu.safetensors', # 肺炎 2分类
    'diagnosis/weights/OAI.safetensors', # 膝盖关节炎 5分类
    # 'diagnosis/weights/Derma.safetensors', # 皮肤病变 7分类
    # 'diagnosis/weights/OCT.safetensors', # 视网膜疾病 4分类
    # 'diagnosis/weights/Path.safetensors', # 结直肠癌的组织病理类型 9分类
    # 'diagnosis/weights/Retina.safetensors', # 糖尿病视网膜病变的严重程度 5分类
    # 'diagnosis/weights/INBreast.safetensors', # 乳腺癌 2分类
]

def model_load(device="cuda"):
    model = timm.create_model("vit_base_patch16_224", pretrained=True, pretrained_cfg_overlay=dict(file=r"C:\Users\DELL\.cache\huggingface\hub\models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k\snapshots\063c6c38a5d8510b2e57df480445e94b231dad2c\model.safetensors"))
    melo = LoRA_ViT_timm_x(model, melo_paths)
    melo = melo.to(device)
    return melo

def illness_define(ImageType):
    if ImageType not in ['Chest X-ray', 'Dermatoscopic', 'Knee X-ray', 'Optical coherence tomography', 'Histopathologic', 'Retinal fundus', 'Mammography']:
        raise ValueError("The illness is not supported.")
    if ImageType == 'Chest X-ray':
        return {0: 'Normal', 1: 'Pneumonia'}
    elif ImageType == 'Dermatoscopic':
        return {0: 'Actinic keratoses and intraepithelial carcinoma', 1: 'Basal cell carcinoma', 2: 'Benign keratosis-like lesions', 3: 'Dermatofibroma', 4: 'Melanoma', 5: 'Melanocytic nevi', 6: 'Vascular lesion'}
    elif ImageType == 'Knee X-ray':
        return {0: 'Normal', 1: 'Doubtful knee osteoarthritis', 2: 'Mild knee osteoarthritis', 3: 'Moderate knee osteoarthritis', 4: 'Severe knee osteoarthritis'}
    elif ImageType == 'Optical coherence tomography':
        return {0: 'Choroidal neovascularization', 1: 'Diabetic macular edema', 2: 'Drusen', 3: 'Normal'}
    elif ImageType == 'Histopathologic':
        return {0: 'Adipose tissue', 1: 'Background', 2: 'Debris', 3: 'Lymphocytes', 4: 'Mucus', 5: 'Smooth muscle', 6: 'Normal colon mucosa', 7: 'Cancer-associated stroma', 8: 'Colorectal adenocarcinoma epithelium'}
    elif ImageType == 'Retinal fundus':
        return {0: 'No abnormalities', 1: 'Mild non-proliferative diabetic retinopathy', 2: 'Moderate non-proliferative diabetic retinopathy', 3: 'Severe non-proliferative diabetic retinopathy', 4: 'Proliferative diabetic retinopathy'}
    elif ImageType == 'Mammography':
        return {0: 'Non-malignant', 1: 'Malignant'}

fivedisease={
        "Cardiomegaly":0,
        "Edema":1,
        "Consolidation":2,
        "Atelectasis":3,
        "Pleural Effusion":4,
           }


from chatcad.cxr.prompt import prob2text
from chatcad.cxr.diagnosis import JFinfer,JFinit
def diagnosis(image, melo, cxr_model, cxr_cfg, ImageType, device="cuda"):
    '''
    image: cv2, BGR,
    melo: model
    ImageType: str
    '''
    types = {'Chest X-ray':[0], 'Knee X-ray':[1],}
    melo.swith_lora(types[ImageType])
    resize= Resize([224,224])
    normalize = Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    medical_image = image


    illness = ''
    with torch.no_grad():
        if ImageType == 'Chest X-ray':
            image = transform(medical_image, cxr_cfg)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            image = image.to(device)
            prob=JFinfer(cxr_model, image, cxr_cfg)
            converter=prob2text(prob, fivedisease)
            diagnosis_result=converter.promptB()
        else: 
            # 输入是BGR
            image = np.array(cv2.cvtColor(medical_image, cv2.COLOR_GRAY2RGB)).astype(np.float32)/ 255.0
            image = rearrange(torch.tensor(image, dtype=torch.float32),'h w c ->c h w')
            image = resize(image)
            image = normalize(image)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            image = image.to(device)
            pred = melo(image)
            prob = list(torch.softmax(pred[0], dim=0).cpu().numpy())
            pred = int(torch.argmax(pred[0], dim=0).cpu().numpy())
            illness_dict = illness_define(ImageType)
            illness = illness_dict[pred]
            diagnosis_result = prob2text__(illness_dict, prob)
    return diagnosis_result, illness, medical_image


def prob2text__(illness_dict, prob):
    diagnosis_result = ''
    for i in range(len(prob)):
        if prob[i] < 0.2:
            diagnosis_result += f"Hardly {illness_dict[i]}. "
        elif prob[i]>=0.2 and prob[i] <0.5:
            diagnosis_result += f"Small possibility of {illness_dict[i]}. "
        elif prob[i]>=0.5 and prob[i] <0.9:
            diagnosis_result += f"Patient is likely to have {illness_dict[i]}. "
        else:
            diagnosis_result += f"Definitely have {illness_dict[i]}. "
    return diagnosis_result

