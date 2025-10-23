import os
import pydicom
from PIL import Image
from tqdm import tqdm
import numpy as np
 
def png_to_dicom(input_png_path, output_dcm_path, patient_name="Anonymous", study_description="PNG to DICOM"):
    for fileNames in os.listdir(input_png_path):
        if fileNames.endswith('.png') == False:
            continue
        input_filename = os.path.basename(fileNames).split('.')[0]
        output_filename = input_filename + ".dcm"
        input_filepath = os.path.join(input_png_path, fileNames)
        output_dcmpath = os.path.join(output_dcm_path, output_filename)
 
        # 读取PNG图像
        img = Image.open(input_filepath)
 
        # 将PNG图像转换为灰度图像（单通道）
        # pixel_array = img.convert("L")

        # 创建一个空的FileDataset对象，并添加DICOM数据集元素
        ds = pydicom.dataset.FileDataset(output_dcm_path, {}, file_meta=pydicom.dataset.Dataset())  # 创建文件元信息头对象
        # 添加DICOM文件元信息头
        ds.file_meta.FileMetaInformationGroupLength = 184
        ds.file_meta.FileMetaInformationVersion = b'\x00\x01'
        ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1.1'
        ds.file_meta.MediaStorageSOPInstanceUID = '1.2.410.200048.2858.20230531153328.1.1.1'
        ds.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'
        ds.file_meta.ImplementationClassUID = '1.2.276.0.7230010.3.0.3.5.4'
        ds.file_meta.ImplementationVersionName = 'ANNET_DCMBK_100'

        pixel_array = np.array(img.getdata(),dtype=np.uint8)
        if img.mode == "L":
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.SamplesPerPixel = 1
        elif img.mode == "RGB":
            ds.PhotometricInterpretation = "RGB"
            ds.SamplesPerPixel = 3

        # 添加DICOM数据集元素
        ds.PatientName = patient_name
        ds.StudyDescription = study_description
        ds.Columns, ds.Rows = img.size
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        # 数据显示格式
        ds.PixelData = pixel_array.tobytes()  # 直接使用灰度图像的字节数据
 
        # 保存DICOM数据集到文件
        ds.is_little_endian = True
        ds.is_implicit_VR = True  # 使用隐式VR
 
        ds.save_as(output_dcmpath)
        print(output_dcmpath)
 

def dcm_uniqueness(input_dcm_path, output_dcm_path):
    # 源文件夹和目标文件夹路径
    source_folder = input_dcm_path
    target_folder = output_dcm_path
    
    patient_pid = 20230726001
    accession_number = 202307261001
    study_uid = 2023072620001
    seriesNumber = 1
    seriesInstanceUID = "1.2.410.200048.2858.20230529094313.1"
    modality = "CR"
    pixelSpacing = [0.160145, 0.160114]
    instanceNumber = 1
    # bodyPartExamined = "CHEST"
    
    # 遍历源文件夹中的文件
    for filename in tqdm(os.listdir(source_folder)):
        if filename.endswith('.dcm'):
            # 构建源文件路径和目标文件路径
            source_file = os.path.join(source_folder, filename)
            target_file = os.path.join(target_folder, filename)
    
            # 加载源DCM文件
            dcm_data = pydicom.dcmread(source_file, force=True)
    
            # 添加患者PID、Accession Number和Study UID等信息
            dcm_data.PatientID = str(patient_pid)
            dcm_data.AccessionNumber = str(accession_number)
            dcm_data.StudyInstanceUID = str(study_uid)
            dcm_data.SeriesNumber = seriesNumber
            dcm_data.SeriesInstanceUID = seriesInstanceUID
            dcm_data.Modality = modality
            dcm_data.PixelSpacing = pixelSpacing
            #dcm_data.BodyPartExamined = bodyPartExamined
            dcm_data.InstanceNumber = instanceNumber
    
            # 将文件名作为患者名
            file_name_without_extension = os.path.splitext(filename)[0]
            dcm_data.PatientName = file_name_without_extension
    
            # 保存修改后的DCM文件到目标文件夹
            dcm_data.save_as(target_file)
    
            # 递增计数器
            patient_pid += 1
            accession_number += 1
            study_uid += 1
        else:
            print("error!")
 
if __name__ == "__main__":
    # 输入PNG图像路径和输出DICOM图像路径
    input_png_path = r"D:\Projects\MedAgent\Image\Pneu_train_png"
    output_dcm_path = r"D:\Projects\MedAgent\Image\DCM_train\Pneu"
 
    # 将PNG转换为DICOM
    png_to_dicom(input_png_path, output_dcm_path)
    dcm_uniqueness(output_dcm_path, output_dcm_path)