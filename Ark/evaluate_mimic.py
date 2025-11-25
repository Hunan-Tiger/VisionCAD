'''
mimic_diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

chexpert_diseases = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

nih14_diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

rsna_diseases = ['No Lung Opacity/Not Normal', 'Normal', 'Lung Opacity']

vindr_diseases = ['Pleural Effusion', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']

shenzhen_diseases = ['TB']
'''
import torch
import torch.nn as nn
import timm.models.swin_transformer as swin
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import math
from sklearn.utils import resample



# rsna_diseases = ['No Lung Opacity/Not Normal', 'Normal', 'Lung Opacity']
mimic_diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

class MIMIC(Dataset):
    def __init__(self, root_path, csv_path):
        self.root_path = root_path
        self.csv_path = csv_path

        self.df = pd.read_csv(self.csv_path)
        self.img_paths = []
        self.labels = []

# root_path 下有两级目录，最后是n张图像。比如：root_path/subject_id/study_id/image0.png, image1.png, ...
# csv_path 是一个csv文件，第一列是subject_id，第二列是study_id，后面是标签
# 根据root_path的所有图像路径，在csv中查询subject_id和study_id，获取对应的标签
        for _, row in tqdm(self.df.iterrows()):
            subject_id = int(row.iloc[0])
            study_id = int(row.iloc[1])
            labels = row.iloc[2:].values
            if subject_id == 19483323 and study_id == 52421170:
                continue  # 跳过这个特定的subject_id和study_id组合
            subject_study_path = os.path.join(self.root_path, str(subject_id), str(study_id))
            
            if os.path.exists(subject_study_path):
                image_files = [f for f in os.listdir(subject_study_path) if f.endswith('.png')]
                for image_file in image_files:
                    self.img_paths.append(os.path.join(subject_study_path, image_file))
                    self.labels.append(labels.astype(str))

        print("Total images:", len(self.img_paths))

        

    def __getitem__(self, index):

        img = Image.open(self.img_paths[index]).convert('RGB')
        img = img.resize((768, 768))
        img = np.array(img) / 255.0
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        img = (img - mean) / std
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        label = self.labels[index]
        label = np.where((label == '-1.0'), '1', label)  # 不确定label视为1，遵循ark+
        label = np.where((label == 'nan') | (label == '0.0') , '0', label).astype(np.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return img, label

    def __len__(self):
        return len(self.img_paths)


class OmniSwinTransformer(swin.SwinTransformer):
    def __init__(self, num_classes_list, projector_features = None, use_mlp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.projector = None 
        if projector_features:
            encoder_features = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = []
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, x, head_n=None):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        # if head_n is not None:
        #     return x, self.omni_heads[head_n](x)
        # else:
        #     return [head(x) for head in self.omni_heads]
        return self.omni_heads[0](x) # MIMIC


    def generate_embeddings(self, x, after_proj = True):
        x = self.forward_features(x)
        if after_proj:
            x = self.projector(x)
        return x




def load_ark(path = 'Ark\Ark6_swinLarge768_ep50.pth.tar'):
    num_classes_list = [14,14,14,3,6,1]
    # Create the base model as before
    model = OmniSwinTransformer(
        num_classes_list,
        projector_features=1376,
        use_mlp=False,
        img_size=768,
        patch_size=4,
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48)
    )

    # Load the checkpoint
    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint['teacher']
    # Remove "module." prefix if present
    if any([True if 'module.' in k else False for k in state_dict.keys()]):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')} 

    # Load the model weights
    msg = model.load_state_dict(state_dict, strict=False)
    print('Loaded with msg:', msg)

    # apply_lora(model, r=16, lora_alpha=16)
    # Freeze all parameters except for LoRA parameters.

    return model