from module_load import module_load, diagnosis_module_load
import time
import cv2
from tools import Tools
import tkinter as tk
import json
import os
from tqdm import tqdm
from PIL import Image
from chat import init_report_generation_chat, add_response_two_image

modules_dict = module_load()
tools = Tools()

API_URL = "https://genaiapi.shanghaitech.edu.cn/api/v1/start"
API_KEY = "Bearer d6df978ccba44a44a7e079e156f326e3"



from openai import OpenAI
client = OpenAI(
    api_key='fk201943-LJ100WLFMnbsQadTjQftWquGwNESrDJz',
    base_url="https://openai.api2d.net/v1"
)


if __name__ == '__main__':

    dcm_list = json.load(open(r'Image\MIMIC\dcm_list.json'))
    annotation = json.load(open(r'Image\MIMIC\annotation.json'))

    imgs_path = r'Image\MIMIC\MIMIC-ezio-kinect-restore'
    imgs = os.listdir(imgs_path)
    iter = 0
    for idx in tqdm(range(1)):
        iter += 1
        img = imgs[idx]
        img_path = os.path.join(imgs_path, img)

        medical_img = cv2.imread(img_path)
        medical_img = cv2.cvtColor(medical_img, cv2.COLOR_BGR2GRAY)
        PartType = tools.identification(medical_img, modules_dict['discrimination_module'][0], modules_dict['discrimination_module'][1], modules_dict['discrimination_module'][2])
        modules_dict = diagnosis_module_load(modules_dict, PartType)

        classifier_results = tools.diagnosis(medical_img, modules_dict['diagnosis_module'][PartType], PartType)

        ref_report = "Patient is status post median sternotomy and CABG. Mild cardiomegaly is similar. The aorta remains tortuous, and the mediastinal and hilar contours are unchanged. Pulmonary vasculature is not engorged. There is minimal atelectasis at the lung bases without focal consolidation. No pleural effusion or pneumothorax is detected. Degenerative changes are seen throughout the thoracic spine."
        ref_medical_img = Image.open(r"D:\Projects\chatcad\ref\02f3df2f-ff2bc640-5f173dca-eaff305d-73b20ae1.jpg").convert('L')
        ref_prompt = 'The diagnostic report for this medical image is as follows:\n'+ref_report

        report_chat = init_report_generation_chat()
        prompt = 'Please use a style similar to the above report to diagnose this medical image. (Do not say anything similar: The diagnostic report for this medical image is as follows)'
        chat_diagnosis = add_response_two_image("user", ref_prompt, prompt, classifier_results, report_chat, ref_medical_img, Image.fromarray(medical_img).convert('L'))
        messages = []
        for role, content in chat_diagnosis:
            messages.append({"role": role, "content": content})

        completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        generated_report = completion.choices[0].message.content

        id = dcm_list[idx]
        gt_report, gt = next((item["report"], item["gt"]) for item in annotation if item["img"] == id)

        result = {
                "id": id,
                "observation": classifier_results,
                "generated_report": generated_report,
                "ground_report": gt_report,
                "gt": gt
            }

        filename = os.path.basename(img_path)
        with open(f'Image\\MIMIC\\{filename}.json', 'a', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            f.write(',')
            f.write('\n')

