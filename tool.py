import glob
from PIL import Image
import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def save_id(file_path,file_output,end):
    file_list = os.listdir(file_path)
    file_id = open(file_output, 'a')
    num=1
    for i in file_list:
        file_id.write(i + ' ')
        if num>=end:
            break
        num+=1
    file_id.close()


def transform_RGB(file_path):
    for filename in glob.glob(file_path):
        img = Image.open(filename).convert('RGB')
        img.save(filename)


if __name__ == '__main__':
    file_path=r'data/CRAG_2/myTraining_Label'
    file_output=r'data_id/img_id86.txt'
    #
    save_id(file_path,file_output,end=86)
    #
    # transform_RGB(r'data/skinlesion/myTraining_Data/*png')
    # transform_RGB(r'data/skinlesion/myTest_Data/*png')








