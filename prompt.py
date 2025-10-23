def get_action_prompt(instruction, width, height, last_action, add_info, error_flag, memory):
    prompt = "### Background ###\n"
    prompt += f"This image is a screenshot of diagnosis software. Its width is {width} pixels and its height is {height} pixels."
    prompt += f"Current The doctor\'s instruction is: {instruction}.\n\n"
    
    prompt += "Please note that this information is not necessarily accurate. You need to combine the screenshot to understand."
    prompt += "\n\n"
    
    if add_info != "":
        prompt += "### Hint ###\n"
        prompt += "There are hints to help you complete the doctor\'s instructions. The hints are as follow:\n"
        prompt += add_info
        prompt += "\n\n"
    
    if memory != "":
        prompt += "### Memory ###\n"
        prompt += "Before predict next action, you DO NOT need to see the slices that have been seen again and can utilize information of the previous one action to reflect on whether your operation were correct. Below, your colleague will tell you which layers need to see and tell your previous one action:"
        prompt += "\n" + memory + "\n"
    
    if error_flag:
        prompt += "### Last operation ###\n"
        prompt += f"You previously wanted to executed the Action: \"{last_action}\"." 
        prompt += "But you find that this operation does not meet your expectation or there's still work to be done. This time you need to reflect and revise or continue to refine your operation."
        prompt += "\n\n"
    
    prompt += "### Response requirements ###\n"
    prompt += "Now you need to combine all of the above to perform just one action on the current status. You must choose one of the three actions below:\n"
    prompt += "- 'SCROLL UP [k]': You can scroll up k layers. Note that you are scrolling through k layers instead of scrolling to k-th layer. For example: Current layer is 7, if you want to see layer 5, you need to scroll up 2 layers."
    prompt += "- 'SCROLL DOWN [k]': You can scroll down k layers. Note that you are scrolling through k layers instead of scrolling to k-th layer. For example: Current layer is 7, if you want to see layer 10, you need to scroll down 3 layers."
    prompt += "- 'Finish': If you think you have seen ALL the layers (For example: all layers is 50, then you need to look at layers 1 to 50.), you can choose this action to terminate the operation process."
    prompt += "\n\n"

    # prompt += "### Brain Position ###\n"
    # prompt += "From the screenshot, identify the positions of the brain in the view requested by the doctor (Axial, Sagittal, and Coronal) and extract its proportional coordinates relative to the entire screenshot (ranging from 0 to 1). It's better to point the center of the brain in a plane."
    # prompt += f"The dimensions of the screenshot are {width} pixels in width and {height} pixels in height."
    # prompt += "The top-left corner of screenshot serves as the origin. The horizontal coordinate is represented by the X-axis, and the vertical coordinate by the Y-axis."
    # prompt += "Note that the regions of the Axial, Sagittal, and Coronal views need to be determined based on the content of the image to accurately locate the center of the brain in each view."
    # prompt += "Hint: First you need to determine the approximate position of view requested (e.g. axial is generally in the top-left part in the screenshot, sagittal is generally in the top-right in the screenshot and coronal is in the bottom-right), then find brain in the center of view."
    # prompt += "The values of x and y are both [0.1-0.4] and [0.1-0.4] on top-left part, [0.6-0.9] and [0.1-0.4] on top-right part, [0.6-0.9] and [0.6-0.9] on bottom-right part."
    # prompt += "Finally, output the **proportional coordinates** in JSON format, for example: {\"axial\": {\"x\": 0.31, \"y\": 0.33}}. Ensure the calculations are precise and reflect the actual layout of the image.\n"
    
    prompt += "### Example for find position\n"
    prompt += "Given a picture the same as the screenshot I give you. And I want to go through axial plane."
    prompt += "1. Find the correct plane: I want to know the position of different plane. And I find that the top-left part is Axial, the top-right part is Sagittal, and the bottom-right part is Coronal. So I choose the top-left part."
    prompt += "2. Find the range of the plane: I find the plane range by finding the black background which is in (x, y, range_x, range_y) means starting from (x,y) and the width is range_x and the height is range_y. Since the screen size is (2560,1440), so the relative position is (x/2560, y/1440, x_range/2560, y_range/1440)."
    prompt += "3. Find the center of the plane: The center is (x-range_x/2,y-range_y/2)."
    prompt += "4. Find current layer: I find the current layer is current 129/150(layer/total layer)."
    prompt += "5. Take action: I click (x-range_x/2,y-range_y/2), and I know my pervious action is 'SCROLL UP [1]' and I have seen layer 130-227 (means I need to see 128 next).So I output 'View: Axial, Layer: 129/227, CLICK [x-range_x/2,y-range_y/2], SCROLL UP [1]'."

    prompt += "### Exact number of current layer ###\n"
    prompt += "You need to find the EXACT number of current layer in the current view. If necessary, it is recommended that you use OCR tool multiple times for ACCURATE identification. **This is very important.**"
    prompt += "Only the information around the current view area is concerned. The number of layers in the current view is typically displayed at the bottom right corner of the blak background."
    prompt += "For example: current view is Axial plane, so I concern the top-left part of the screenshot and find the number of layers in the bottom right corner of the top-left black background."
    prompt += "The number of other view layers DOES NOT need attention and MUST NOT be confused."

    prompt += "### Output format ###\n"
    prompt += "Your output consists of the following three parts:\n"
    prompt += "### Thought ###\nThink about control the plane area you want to view by clicking on the mouse and think about how many layers you need to scroll from the current layer to a layer that has not yet been observed.\n"
    prompt += "### Action ###\nYou output 1.view name needed and 2. the number of layers on current view plane and 3. click position (result of Brain Position) and 4. only choose one from the three actions above. EXAMPLE: 'View: Coronal, Current Layer: 129/227, CLICK [0.30, 0.29], SCRLL UP [1]'(MAKE SURE the output format is consistent with example)\n"
    prompt += "### Operation ###\nPlease generate a brief natural language description for the operation in Action based on your Thought."
    
    return prompt


def get_reflect_prompt(instruction, clickable_infos1, clickable_infos2, width, height, keyboard1, keyboard2, summary, action, add_info):
    prompt = f"These images are two phone screenshots before and after an operation. Their widths are {width} pixels and their heights are {height} pixels.\n\n"
    
    prompt += "In order to help you better perceive the content in this screenshot, we extract some information on the current screenshot through system files. "
    prompt += "The information consists of two parts, consisting of format: coordinates; content. "
    prompt += "The format of the coordinates is [x, y], x is the pixel from left to right and y is the pixel from top to bottom; the content is a text or an icon description respectively "
    prompt += "The keyboard status is whether the keyboard of the current page is activated."
    prompt += "\n\n"
    
    prompt += "### Before the current operation ###\n"
    prompt += "Screenshot information:\n"
    for clickable_info in clickable_infos1:
        if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
            prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
    prompt += "Keyboard status:\n"
    if keyboard1:
        prompt += f"The keyboard has been activated."
    else:
        prompt += "The keyboard has not been activated."
    prompt += "\n\n"

    prompt += "### After the current operation ###\n"
    prompt += "Screenshot information:\n"
    for clickable_info in clickable_infos2:
        if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info['coordinates'] != (0, 0):
            prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
    prompt += "Keyboard status:\n"
    if keyboard2:
        prompt += f"The keyboard has been activated."
    else:
        prompt += "The keyboard has not been activated."
    prompt += "\n\n"
    
    prompt += "### Current operation ###\n"
    prompt += f"The user\'s instruction is: {instruction}. You also need to note the following requirements: {add_info}. In the process of completing the requirements of instruction, an operation is performed on the phone. Below are the details of this operation:\n"
    prompt += "Operation thought: " + summary.split(" to ")[0].strip() + "\n"
    prompt += "Operation action: " + action
    prompt += "\n\n"
    
    prompt += "### Response requirements ###\n"
    prompt += "Now you need to output the following content based on the screenshots before and after the current operation:\n"
    prompt += "Whether the result of the \"Operation action\" meets your expectation of \"Operation thought\"?\n"
    prompt += "A: The result of the \"Operation action\" meets my expectation of \"Operation thought\".\n"
    prompt += "B: The \"Operation action\" results in a wrong page and I need to return to the previous page.\n"
    prompt += "C: The \"Operation action\" produces no changes."
    prompt += "\n\n"
    
    prompt += "### Output format ###\n"
    prompt += "Your output format is:\n"
    prompt += "### Thought ###\nYour thought about the question\n"
    prompt += "### Answer ###\nA or B or C"
    
    return prompt


def get_memory_prompt(insight):
    if insight != "":
        prompt  = "### Important content ###\n"
        prompt += insight
        prompt += "\n\n"
    
        prompt += "### Response requirements ###\n"
        prompt += "Please think about whether there is any content closely related to ### Important content ### on the current page? If there is, please output the content. If not, please output \"None\".\n\n"
    
    else:
        prompt  = "### Response requirements ###\n"
        prompt += "Please think about whether there is any content closely related to user\'s instrcution on the current page? If there is, please output the content. If not, please output \"None\".\n\n"
    
    prompt += "### Output format ###\n"
    prompt += "Your output format is:\n"
    prompt += "### Important content ###\nThe content or None. Please do not repeatedly output the information in ### Memory ###."
    
    return prompt

def get_click_prompt(width, height, view):
    prompt = f"Please provide the coordinates (x, y) for specific {view} plane in this {width} by {height} screenshot. Return only the coordinates as numbers, nothing else."
    return prompt


# for chest xray
def get_perception_prompt(instruction, width, height, add_info):
    prompt = "### Background ###\n"
    prompt += f"This is a picture of a computer screen taken by a video camera. Its width is {width} pixels and its height is {height} pixels."
    if instruction != "":
        prompt += f"Current the doctor\'s instruction is: {instruction}.\n\n"
    
    prompt += "A computer monitor that is displaying medical images is the subject in this picture.\n"

    if add_info != "":
        prompt += "### Hint ###\n"
        prompt += "There are hints to help you complete the doctor\'s instructions. The hints are as follow:\n"
        prompt += add_info
        prompt += "\n\n"

    prompt += "\n### What types of medical images are identified ###\n"
    prompt += "Such as Knee_CT, Chest_X-ray, etc. You need to identify the type of medical image in the picture.\n"

    # prompt += "### Locate the location of medical image ###\n"
    # prompt += "Analyze the given medical image displayed on the screen."
    # prompt += "Identify three specific points or positions within the image that are of interest, and provide their precise normalized coordinates as ratios within the range of 0 to 1."
    # prompt += "These ratios should be relative to the width and height of the visible image, where (0,0) corresponds to the top-left corner and (1,1) corresponds to the bottom-right corner."
    # prompt += "Output the result as a list of three 2D coordinates in the format: [[x1, y1], [x2, y2], [x3, y3]].\n"
    prompt +='''
### Locate the location of medical image ###\n
You can see one or more medical images (e.g., X-ray images). Your task is:
1. For each medical image, select exactly two points that lie within the meaningful content area of the image (As much as possible in the center of the medical image). 
Make sure that these points are not placed outside the boundaries of the image or in non-informative (e.g., black border) regions.
2. Report the coordinates of these points as normalized ratios relative to the displayed image dimensions:
   - (0,0) corresponds to the top-left corner.
   - (1,1) corresponds to the bottom-right corner.
   All coordinates must be floating-point numbers between 0 and 1.
3. Output Format: If there are N images, you must output 2N points in total. The final output should be a list of coordinates in the following format:
   [[x11, y11], [x12, y12], [x21, y21], [x22, y22], ... ]
   Here, [x11, y11] and [x12, y12] are the two points for the first image, [x21, y21] and [x22, y22] are for the second image, and so on.
4. Ensure that each chosen point is clearly within the actual image region and not on the black border or outside the image's meaningful content.
5. Make sure to return all coordinates with two decimal places, strictly within the range [0, 1].

For example (for formatting only, not actual points):
If there are 3 images, return 6 points like this:
[[0.23, 0.45], [0.56, 0.72], [0.12, 0.89], [0.67, 0.34], [0.42, 0.11], [0.98, 0.56]]
    '''

    # prompt += "### Locate the position of computer monitor ###\n"
    # prompt += "You need to locate the computer monitor in this picture.\n"
    # prompt += "You need to give the scale coordinates of 6 points, and if visualized, these points will defineitely be INSIDE the region of computer monitor, instead of the corner."
    # prompt += "The output of these points is formatted as a two-dimensional list: [[x1, y1], [x2, y2], ..., [x6, y6]] \n"

    prompt += "\n### Output format ###\n"
    prompt += "Your output format is as follows:"
    prompt += "Points-[[x11, y11], [x12, y12], ... ]; Type-[medical image type]. For example: Points-[[0.27, 0.33], [0.27, 0.35], ... ]; Type-Chest_X-ray\n" # For example: Points-[[0.32, 0.51], [0.41, 0.63], [0.54, 0.61]]; Type-Chest_X-ray
    # prompt += "At last, you need to explain the reason for your choice of these points."
    return prompt

def get_diagnosis_prompt(diagnosis_result):
    prompt = "\n### Analize Medical Image ###\n"
    prompt += "Please analyze the medical image to assist the doctor to write analysis report based on the doctor's diagnosis, "
    prompt += "and finally give your report.\n"
    prompt += "\n\n"
    prompt += f"\nThe doctor's diagnosis is: {{{diagnosis_result}}}\n"

    # prompt += "\n### Example Output ###\n"
    # prompt += "Your output format should be similar: The chest X-ray shows areas of increased opacity in the lung fields, which would suggest pneumonia. The heart size appears normal, but there is evidence of possible consolidation in the lower lobes.\n"
    return prompt

def get_identify_prompt():
    prompt = "\n### Identify medical image types ###\n"
    prompt += "Such as Chest_X-ray, Dermatoscopic_Image, Knee_X-ray, Optical_coherence_tomography_Image, Histopathologic_Image, Retinal_fundus_Image, Mammography_Image, etc. You need to determine which of these types this medical image is.\n"
    prompt += "### Output format ###\n"
    prompt += "Your output format is as follows: Dermatoscopic_Image"
    return prompt