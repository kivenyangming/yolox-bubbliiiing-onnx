import random
import time
from math import exp, pow
import cv2
import numpy as np
import onnxruntime as ort
from numba import jit

className = ["red", "green", "yellow", "black"]
# 替换对应yolo的Anchors值


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
    hw = np.asarray([[20, 20], [40, 40], [80, 80]], dtype=np.int32)
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
    grids, strides = get_grids_strides()
    pdata[:, :, 4:] = Sigmoid(pdata[:, :, 4:])
    pdata[..., :2] = (pdata[..., :2] + grids) * strides
    pdata[..., 2:4] = np.exp(pdata[..., 2:4]) * strides

    # -----------------#
    #   归一化
    # -----------------#
    pdata[..., [0, 2]] = pdata[..., [0, 2]] / netWidth
    pdata[..., [1, 3]] = pdata[..., [1, 3]] / netHeight

    box_corner = pdata.copy()

    box_corner[:, :, 0] = pdata[:, :, 0] - pdata[:, :, 2] / 2
    box_corner[:, :, 1] = pdata[:, :, 1] - pdata[:, :, 3] / 2
    box_corner[:, :, 2] = pdata[:, :, 0] + pdata[:, :, 2] / 2
    box_corner[:, :, 3] = pdata[:, :, 1] + pdata[:, :, 3] / 2
    pdata[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(pdata))]

    for i, image_pred in enumerate(pdata):
        # ----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        # ----------------------------------------------------------#

        class_conf = np.max(image_pred[:, 5:5 + 80], axis=1, keepdims=True)
        class_pred = np.argmax(image_pred[:, 5:5 + 80], axis=1)
        class_pred = class_pred.reshape(8400, 1)

        #   利用置信度进行第一轮筛选
        conf_mask = np.squeeze((image_pred[:, 4] * class_conf[:, 0] >= 0.5))

        if not image_pred.shape[0]:
            continue

        detections = np.concatenate((image_pred[:, :5], class_conf, class_pred.astype(float)), axis=1)
        detections = detections[conf_mask]
        # cv2.dnn.NMSBoxes的bboxes框应该是左上角坐标(x,y)和 w，h， 那么针对不同模型的，要具体情况转化一下，才能应用该函数。
        boxes = detections[:, :4]
        confidences = detections[:, 4] * detections[:, 5]

        nms_result = cv2.dnn.NMSBoxes(boxes, confidences, classThreshold, nmsThreshold)  # 抑制处理 返回的是一个数组
        if len(nms_result) == 0:
            return
        else:
            for idx in nms_result:
                print(idx)
                # classIdx = class_pred[idx]
                # confidenceIdx = confidences[idx]
                # boxsIdx = boxes[idx]
                #
                # pt1 = (boxsIdx[0], boxsIdx[1])
                # pt2 = (boxsIdx[0] + boxsIdx[2], boxsIdx[1] + boxsIdx[3])
                # # 绘制图像目标位置
                # # x1, y1, x2, y2 = box[0], box[1], box[0] + box[2], box[1] + box[3]
                # cv2.rectangle(SrcImg, pt1, pt2, classIdx, 2, 2)
                # cv2.rectangle(SrcImg, (boxsIdx[0], boxsIdx[1] - 18), (boxsIdx[0] + boxsIdx[2], boxsIdx[1]), (255, 0, 255),
                #               -1)
                # label = "%s:%s" % (className[classIdx], confidenceIdx)  # 给目标进行添加类别名称以及置信度
                # FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
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

