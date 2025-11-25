import base64
import requests
from PIL import Image
import io
import time
import json


def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def inference_chat(chat, API_URL, API_KEY, model="gpt-4.1"):    
    headers = {
        "Content-Type": "application/json",
        "Authorization": API_KEY
    }
    messages = []
    for role, content in chat:
        messages.append({"role": role, "content": content})
    data = {
        'model': model,
        "messages": messages,
        "max_tokens": 512,
        'temperature': 0.2,
    }
    # data = json.dumps({
    #     'model': model,
    #     "messages": messages,
    #     "max_tokens": 1000,
    #     'temperature': 0.0,
    # })

    try:
        res = requests.post(API_URL, headers=headers, json=data)
        #res = requests.request("POST", API_URL, headers=headers, data=data)
        res_json = res.json()
        res_content = res_json['choices'][0]['message']['content']

        # 获取 token 使用情况
        usage = res_json.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)

        # print("Current Time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(f"花费 token 数: {total_tokens}")
    except:
            print("Network Error:")
            try:
                print(res_json)
            except:
                print("Inference Failed")
    
    return res_content


def memory_chat(memory, API_URL, API_KEY):
    headers = {
        "Content-Type": "application/json",
        "Authorization": API_KEY
    }

    prompt = "You are a helpful medical image analysis assistant. You need help your colleague to remember these slice layers that have been observaed and SIMPLY tell him which layers aren't being seen."
    prompt += "Sometimes your colleague may make mistake of the total layers, so you need to decide which number he/she gives you is more reasonable. "
    prompt += "For example, you got this information - Current slices of the axial plane has 10 layers, layers that have been seen: [1, 2, 3, 4, 5, 6], so you should output: Total_layer: 10\n Layer_not_seen: [7, 8, 9, 10]."
    prompt += "You MUST output exactly as the Output Format: Total_layer: [total_layer]\n Layer_not_seen: [layer numbers]. "

    message = [
        {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text":  prompt
                    }
                        ]
                    },
                {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"You have remembered these information: {memory}\n\n You need to tell your colleague which layers aren't being seen."
                    }
                    
                ]
            }
        ]
    data = {
        "messages": message,
        "max_tokens": 1000,
        'temperature': 0.0,
    }    

    try:
        res = requests.post(API_URL, headers=headers, json=data)
        res_json = res.json()
        response_text = res_json['choices'][0]['message']['content']
    except:
        print("Network Error:")
        try:
            print(res_json)
        except:
            print("Memory Failed")

    return response_text