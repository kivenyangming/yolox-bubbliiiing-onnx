import random
import time
from math import exp, pow
import cv2
import numpy as np
import onnxruntime as ort
from numba import jit

className = list(map(lambda x: x.strip(), open('coco.names', 'r').readlines()))  # 可替换自己的类别文件

netStride = np.asarray([8.0, 16.0, 32.0], dtype=np.float32)
netWidth = 640
netHeight = 640
nmsThreshold = 0.50
boxThreshold = 0.50
classThreshold = 0.50


def GetColors(color_num):
    ColorList = []
    for num in range(color_num):
        R = random.randint(0, 255)
        G = random.randint(0, 255)
        B = random.randint(0, 255)
        BGR = (B, G, R)
        ColorList.append(BGR)
    return ColorList


#
@jit
def Sigmoid(x):
    # x = float(x)
    # out = max(x, 0)
    out = (float(1.) / (float(1.) + np.exp(-x)))
    return out


def readModel(netPath):
    net = ort.InferenceSession(netPath, providers=ort.get_available_providers())
    return net

def get_grids_strides():
    grids = []
    strides = []
    hw = np.asarray([[80, 80], [40, 40], [20, 20]], dtype=np.float32)
    for h, w in hw:
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w))
        grid = np.stack((grid_x, grid_y), 2)
        grid.shape = (1, -1, 2)
        shape = grid.shape[:2]

        grids.append(grid)
        strides.append(np.full((shape[0], shape[1], 1), netWidth / h))

    grids = np.concatenate(grids, axis=1).astype(np.float32)
    strides = np.concatenate(strides, axis=1).astype(np.float32)
    return grids, strides


def Detect(SrcImg, net, netWidth, netHeight):

    netInputImg = SrcImg
    blob = cv2.dnn.blobFromImage(netInputImg, scalefactor=1 / 255.0, size=(netWidth, netHeight), mean=[104, 117, 123],
                                 swapRB=True, crop=False)

    input_name = "images"
    output_name = "output"
    netOutputImg = net.run([output_name], {input_name: blob.astype(np.float32)})
    pdata = netOutputImg[0]
    classIds = []  # 定义结果线性表
    confidences = []  # 定义置信度线性表
    boxes = []  # 定义坐标线性表
    count = 0

    for stride in range(3):  # netStride = {8.0, 16.0, 32.0} = 3
        grid_x = netWidth / netStride[stride]
        grid_y = netHeight / netStride[stride]
        grid_x, grid_y = int(grid_x), int(grid_y)  # 系统默认是float32，这里是为了下面的循环转为int
        for i in range(grid_x):
            for j in range(grid_y):
                pdatabox = pdata[0][count][4]
                box_score = Sigmoid(pdatabox)  # 获取每一行的box框中含有某个物体的概率
                if box_score > boxThreshold:  # box的阈值起作用了
                    scores = pdata[0][count][5:]  # 这里的scores理应是一个多维矩阵
                    scores = Sigmoid(scores)
                    _, max_class_socre, _, classIdPoint = cv2.minMaxLoc(scores)  # 求最大值以及最大值的位置&位置是元组
                    # max_class_socre = np.asarray(max_class_socre, dtype=np.float64)
                    # max_class_socre = Sigmoid(max_class_socre)
                    if max_class_socre > classThreshold:  # 类别的置信度起作用
                        strides = netStride[stride]
                        pdatax = pdata[0][count][0]
                        x1 = (pdatax + j) * strides  # xmin
                        pdatay = pdata[0][count][1]
                        y1 = (pdatay + i) * strides  # ymin
                        pdataw = pdata[0][count][2]
                        x2 = np.exp(pdataw) * strides  # xmax
                        pdatah = pdata[0][count][3]
                        y2 = np.exp(pdatah) * strides  # ymax

                        x = x1 - (x2 / 2)
                        y = y1 - (y2 / 2)
                        # w = x1 - x2 / 2
                        # h = y1 - y2 / 2
# ---------------------------------------------------------------------------------------------------------- #
                        left, top, W, H = int(x), int(y), int(x2), int(y2)
                        # 对classIds & confidences & boxes
                        classIds.append(classIdPoint[1])  # 获取最大值的位置
                        confidences.append(max_class_socre * box_score)
                        boxes.append((left, top, W, H))
                count += 1
    # cv2.dnn.NMSBoxes的bboxes框应该是左上角坐标(x,y)和 w，h， 那么针对不同模型的，要具体情况转化一下，才能应用该函数。
    nms_result = cv2.dnn.NMSBoxes(boxes, confidences, classThreshold, nmsThreshold)  # 抑制处理 返回的是一个数组
    if len(nms_result) == 0:
        return
    else:
        for idx in nms_result:
            classIdx = classIds[idx]
            confidenceIdx = confidences[idx]
            boxsIdx = boxes[idx]

            pt1 = (boxsIdx[0], boxsIdx[1])
            pt2 = (boxsIdx[0] + boxsIdx[2], boxsIdx[1] + boxsIdx[3])
            # 绘制图像目标位置
            # x1, y1, x2, y2 = box[0], box[1], box[0] + box[2], box[1] + box[3]
            cv2.rectangle(SrcImg, pt1, pt2, classIdx, 2, 2)
            cv2.rectangle(SrcImg, (boxsIdx[0], boxsIdx[1] - 18), (boxsIdx[0] + boxsIdx[2], boxsIdx[1]), (255, 0, 255),
                          -1)
            label = "%s:%s" % (className[classIdx], confidenceIdx)  # 给目标进行添加类别名称以及置信度
            FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(SrcImg, label, (boxsIdx[0] - 2, boxsIdx[1] - 5), FONT_FACE, 0.5, (0, 0, 0), 1)


if __name__ == "__main__":
    model_path = "yolox_nano.onnx"
    classNum = 80
    Mynet = readModel(model_path)
    frame = cv2.imread("street.jpg")
    frame = cv2.resize(frame, (640, 640))
    Detect(frame, Mynet, netWidth, netHeight)

    cv2.imshow("video", frame)
    cv2.waitKey(0)

