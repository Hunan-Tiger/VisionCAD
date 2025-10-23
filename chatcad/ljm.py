from cxr.prompt import prob2text
from cxr.diagnosis import getJFImg,JFinfer,JFinit
import json

fivedisease={
        "Cardiomegaly":0,
        "Edema":1,
        "Consolidation":2,
        "Atelectasis":3,
        "Pleural Effusion":4,
           }

class Classifier():
    def __init__(self, img_model, imgcfg):
        self.img_model = img_model
        self.imgcfg = imgcfg

    def report_cxr(self, img_path):
        #img=img # 修正后的灰度图像
        img1,img2=getJFImg(img_path, self.imgcfg)
        prob=JFinfer(self.img_model, img2, self.imgcfg)
        converter=prob2text(prob, fivedisease)
        # default setting: promptB
        res=converter.promptB()
        return res

img_path = r"D:\Projects\chatcad\ref\02f3df2f-ff2bc640-5f173dca-eaff305d-73b20ae1.jpg"
c = Classifier()
a = c.report_cxr(img_path=img_path)