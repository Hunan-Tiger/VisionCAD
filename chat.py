import copy
from api import encode_image

def init_perception_chat():
    perception_chat = []
    sysetm_prompt = "You are a specialized medical image analysis assistant working with doctor to analyze medical images systematically."
    perception_chat.append(["system", [{"type": "text", "text": sysetm_prompt}]])
    return perception_chat

def init_diagnosis_chat():
    diagnosis_chat = []
    sysetm_prompt = "You are a useful medical assistant and you need to take up a role as an assistant who helps doctors to diagnose medical images."
    diagnosis_chat.append(["system", [{"type": "text", "text": sysetm_prompt}]])
    return diagnosis_chat

def init_restore_chat():
    restore_chat = []
    sysetm_prompt = """You are a medical image analysis expert. Analyze the provided medical image and generate a simple prompt that describes:
1. Modality (e.g., MRI, CT, X-ray)
2. Anatomical region (e.g., brain, knee, abdomen)
3. View/plane (e.g., axial, coronal, sagittal)
4. Image quality issues:
    - Artifacts (e.g., motion artifacts, metal artifacts)
    - Exposure problems (over/underexposure)
    - Noise levels
    - Resolution issues
    - Contrast problems
    - Edge definition
    - Any missing or truncated regions
5. Tissue visibility and contrast
6. Any visible anatomical structures and their clarity
7. Scanning parameters if visible (e.g., slice thickness, field of view)

Format your response strictly as:
Modality: {modality}
Region: {anatomical_region}
View: {view}
Quality Issues (If don't have the issue listed below, delete it. If there exists issue not listed below, add it.):
- Artifacts: {list specific artifacts}
- Exposure: {describe exposure problems}
- Noise: {noise level}
- Resolution: {resolution quality}
- Contrast: {contrast issues}
Structures Affected: {list affected anatomical structures}
Required Improvements: {list specific improvements needed}

Keep descriptions brief and focused on key issues. DONT EXCEED 70 WORDS. SO KEEP THE MOST IMPORTANT INFORMATION ONLY."""
    restore_chat.append(["system", [{"type": "text", "text": sysetm_prompt}]])
    return restore_chat

def init_report_generation_chat():
    report_generation_chat = []
    sysetm_prompt = "You are a specialized medical image analysis assistant working with doctor to generate diagnostic reports."
    report_generation_chat.append(["system", [{"type": "text", "text": sysetm_prompt}]])
    return report_generation_chat

def add_response(role, prompt, chat_history, image=None):
    new_chat_history = copy.deepcopy(chat_history)
    if image is not None:
        base64_image = encode_image(image)
        content = [
            {
                "type": "text", 
                "text": prompt
            },
            {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            },
        ]
    else:
        content = [
            {
            "type": "text", 
            "text": prompt
            },
        ]
    new_chat_history.append([role, content])
    return new_chat_history

def add_response_restore(role, chat_history, image):
    new_chat_history = copy.deepcopy(chat_history)
    base64_image = encode_image(image)
    content = [
        {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        },
    ]

    new_chat_history.append([role, content])
    return new_chat_history


def add_response_two_image(role, ref_prompt, prompt, diagnosis_hint, chat_history, image1, image2):
    new_chat_history = copy.deepcopy(chat_history)

    base64_image1 = encode_image(image1)
    base64_image2 = encode_image(image2)
    content = [
        {
            "type": "text", 
            "text": ref_prompt
        },
        {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image1}"
            }
        },
        {
            "type": "text", 
            "text": prompt
        },
        {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image2}"
            }
        },
        {
            "type": "text", 
            "text": 'Hint: The diagnostic results of the medical image diagnostic model are provided for your reference——\n' + diagnosis_hint
        },
    ]

    new_chat_history.append([role, content])
    return new_chat_history


def print_status(chat_history):
    print("*"*100)
    for chat in chat_history:
        print("role:", chat[0])
        print(chat[1][0]["text"] + "<image>"*(len(chat[1])-1) + "\n")
    print("*"*100)