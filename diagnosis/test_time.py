import timm
import torch
import time
from lora import LoRA_ViT_timm_x
import copy
import logging
import random


# save log to file
logging.basicConfig(level=logging.INFO, filename='test_time.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.info("test")

melo_paths = [
    'diagnosis/weights/Pneu.safetensors', # 肺炎 2分类
    'diagnosis/weights/Derma.safetensors', # 皮肤病变 7分类
    'diagnosis/weights/OAI.safetensors', # 膝盖关节炎 5分类
    'diagnosis/weights/OCT.safetensors', # 视网膜疾病 4分类
    'diagnosis/weights/Path.safetensors', # 结直肠癌的组织病理类型 9分类
    'diagnosis/weights/Retina.safetensors', # 糖尿病视网膜病变的严重程度 5分类
]

batch_size_list = [1, 2, 4, 8, 16, 32]

# test melo
for melo_number in [1, 2, 4, 8]:
    temp_melo_paths = melo_paths[:melo_number]
    for batch_size in batch_size_list:
        switch_time_list = []
        test_time_list = []
        model = timm.create_model("vit_base_patch16_224", pretrained=True).cuda()
        melo = LoRA_ViT_timm_x(model, temp_melo_paths).cuda()
        batch_img = torch.randn(batch_size, 3, 224, 224).cuda()
        for _ in range(5):
            switch_time = 0
            start_time = time.time()
            for i in range(256 // batch_size):
                # 随机生成n*b的list，b为batch_size, 值为0~3
                batch_index = torch.randint(0, len(temp_melo_paths), (batch_size,))
                switch_time_start = time.time()
                melo.swith_lora(batch_index)
                switch_time += time.time() - switch_time_start
                with torch.no_grad():
                    out = melo(batch_img)
                
            end_time = time.time()
            # print("switch time: ", switch_time)
            # print("test time: ", end_time - start_time)
            
            switch_time_list.append(switch_time)
            test_time_list.append(end_time - start_time)

        # mean and std
        logging.info("############################################################")
        logging.info(f"melo_number: {melo_number}, batch_size: {batch_size}")
        logging.info(f"switch average: {sum(switch_time_list) / len(switch_time_list)}")
        logging.info(f"switch std: {sum([(x - sum(switch_time_list) / len(switch_time_list)) ** 2 for x in switch_time_list]) / len(switch_time_list)}")
        logging.info(f"test average: {sum(test_time_list) / len(test_time_list)}")
        logging.info(f"test std: {sum([(x - sum(test_time_list) / len(test_time_list)) ** 2 for x in test_time_list]) / len(test_time_list)}")


# test fine-tune
# one task
# model = timm.create_model("vit_gigantic_patch14_clip_224.laion2b", pretrained=True).cuda()
# for melo_number in [8]:
#     batch_size = 1
#     batch_img = torch.randn(batch_size, 3, 224, 224).cuda()
#     switch_time_list = []
#     test_time_list = []
#     for _ in range(1):
#         switch_time = 0
#         start_time = time.time()
#         previouse_index = -1
#         for i in range(256 // batch_size):
#             current_index = random.randint(0, melo_number - 1)
#             if current_index != previouse_index:
#                 switch_time_start = time.time()
#                 del model
#                 torch.cuda.empty_cache()
#                 model = timm.create_model("vit_gigantic_patch14_clip_224.laion2b", pretrained=True).cuda()
#                 switch_time += time.time() - switch_time_start
#                 previouse_index = current_index

#             with torch.no_grad():
#                 out = model(batch_img)

#         end_time = time.time()
#         # print("switch time: ", switch_time)
#         # print("test time: ", end_time - start_time)

#         switch_time_list.append(switch_time)
#         test_time_list.append(end_time - start_time)
        
#     # mean and std
#     logging.info("############################################################")
#     logging.info(f"melo_number: {melo_number}")
#     logging.info(f"switch average: {sum(switch_time_list) / len(switch_time_list)}")
#     logging.info(f"switch std: {sum([(x - sum(switch_time_list) / len(switch_time_list)) ** 2 for x in switch_time_list]) / len(switch_time_list)}")
#     logging.info(f"test average: {sum(test_time_list) / len(test_time_list)}")
#     logging.info(f"test std: {sum([(x - sum(test_time_list) / len(test_time_list)) ** 2 for x in test_time_list]) / len(test_time_list)}")
