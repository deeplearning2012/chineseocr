import os
import json
import time
import web
import numpy as np
import uuid
from PIL import Image
from config import *
from apphelper.image import union_rbox,adjust_box_to_origin,base64_to_PIL
from application import trainTicket,idcard
import base64
import requests

def read_img_base64(p):
    with open(p,'rb') as f:
        imgString = base64.b64encode(f.read())
    imgString=b'data:image/jpeg;base64,'+imgString
    return imgString.decode()

imgString = read_img_base64('test/idcard-demo.jpeg')
billModel = '身份证'
textAngle = True

from crnn.keys import alphabetChinese, alphabetEnglish

if ocrFlag == 'keras':
    from crnn.network_keras import CRNN

    if chineseModel:
        alphabet = alphabetChinese
        if LSTMFLAG:
            ocrModel = ocrModelKerasLstm
        else:
            ocrModel = ocrModelKerasDense
    else:
        ocrModel = ocrModelKerasEng
        alphabet = alphabetEnglish
        LSTMFLAG = True

elif ocrFlag == 'torch':
    from crnn.network_torch import CRNN

    if chineseModel:
        alphabet = alphabetChinese
        if LSTMFLAG:
            ocrModel = ocrModelTorchLstm
        else:
            ocrModel = ocrModelTorchDense

    else:
        ocrModel = ocrModelTorchEng
        alphabet = alphabetEnglish
        LSTMFLAG = True
elif ocrFlag == 'opencv':
    from crnn.network_dnn import CRNN

    ocrModel = ocrModelOpencv
    alphabet = alphabetChinese
else:
    print("err,ocr engine in keras\opencv\darknet")

nclass = len(alphabet) + 1
if ocrFlag == 'opencv':
    crnn = CRNN(alphabet=alphabet)
else:
    crnn = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=LSTMFLAG, GPU=GPU, alphabet=alphabet)
if os.path.exists(ocrModel):
    crnn.load_weights(ocrModel)
else:
    print("download model or tranform model with tools!")

ocr = crnn.predict_job

if yoloTextFlag=='opencv':
    scale,maxScale = IMGSIZE
    from text.opencv_dnn_detect import text_detect
elif yoloTextFlag=='darknet':
    scale,maxScale = IMGSIZE
    from text.darknet_detect import text_detect
elif yoloTextFlag=='keras':
    scale,maxScale = IMGSIZE[0],2048
    from text.keras_detect import  text_detect
else:
     print( "err,text engine in keras\opencv\darknet")

from text.opencv_dnn_detect import angle_detect

from main import TextOcrModel
model =  TextOcrModel(ocr,text_detect,angle_detect)

imgString = imgString.encode().split(b';base64,')[-1]
img = base64_to_PIL(imgString)
if img is not None:
    img = np.array(img)

H, W = img.shape[:2]


detectAngle = textAngle
result,angle= model.model(img,
                            scale=scale,
                            maxScale=maxScale,
                            detectAngle=detectAngle,##是否进行文字方向检测，通过web传参控制
                            MAX_HORIZONTAL_GAP=100,##字符之间的最大间隔，用于文本行的合并
                            MIN_V_OVERLAPS=0.6,
                            MIN_SIZE_SIM=0.6,
                            TEXT_PROPOSALS_MIN_SCORE=0.1,
                            TEXT_PROPOSALS_NMS_THRESH=0.3,
                            TEXT_LINE_NMS_THRESH = 0.99,##文本行之间测iou值
                            LINE_MIN_SCORE=0.1,
                            leftAdjustAlph=0.01,##对检测的文本行进行向左延伸
                            rightAdjustAlph=0.01,##对检测的文本行进行向右延伸
                           )

print(result)