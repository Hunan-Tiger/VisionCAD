import os
import cv2
import torch
import numpy as np
from .model import Restormer


def restormer(input_img, ckpt_path='restorer/restormer_best_sc.pth', img_size=(512, 512)):
    '''
    input_img: cv2, BGR

    return: 
    out_img: numpy, gray [H, W]
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Restormer(inp_channels=1, out_channels=1, dim=48).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 改为灰度图
    #input_img = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)


    if img_size != None:
    # 调整大小
        input_img = cv2.resize(input_img, img_size)

    # Convert to tensor [H,W] -> [1,1,H,W]
    input_img = torch.FloatTensor(input_img).unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        input_img = input_img.to(device)
        output = model(input_img)

        output = output[0]
        out_np = output.cpu().numpy().squeeze(0)*255
        out_np = out_np.astype(np.float32) * 1.3

        out_np = np.clip(out_np, 0, 255).astype(np.uint8)

        #cv2.imwrite("restorer/restored.png", out_np)

        return out_np 

if __name__ == "__main__":
    restormer("D:\Projects\MedAgent\screenshot\detect\medical_img\img_11.png",'restorer/restormer_best_sc.pth', (512, 512))