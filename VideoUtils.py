import cv2
import numpy as np

def py_cpu_nms(dets, thresh):
    y1 = dets[:, 1]
    x1 = dets[:, 0]
    y2 = y1 + dets[:, 3]
    x2 = x1 + dets[:, 2]

    scores = dets[:, 4]  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]

    # keep为最后保留的边框
    keep = []

    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)

        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]

        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return keep

def calculate(bound, mask):
    x, y, w, h = bound

    area = mask[y:y + h, x:x + w]

    pos = area > 0 + 0

    score = np.sum(pos) / (w * h)

    return score

def nms_cnts( cnts, mask, min_area):

    bounds = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > min_area]

    if len(bounds) == 0:
        return []

    scores = [calculate(b, mask) for b in bounds]

    bounds = np.array(bounds)

    scores = np.expand_dims(np.array(scores), axis=-1)

    keep = py_cpu_nms(np.hstack([bounds, scores]), 0.3)

    return bounds[keep]

def getFrameDiff(videoPath):
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        # 如果没有检测到摄像头，报错
        raise Exception('Check if the camera is on.')
    ret, frame = cap.read()
    prevframe = frame  # 第一帧
    count = 0
    while True:
        ret, frame = cap.read()
        nextframe = frame
        count += 1
        if not ret:
            break
        if ret and count==5:
            count=0
            diff = cv2.absdiff(prevframe, nextframe)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # 将帧差图转换成灰度图
            _, mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
            mask = cv2.medianBlur(mask, 3)  # 中值滤波
            es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.erode(mask, es, iterations=1)  # 腐蚀
            mask = cv2.dilate(mask, es, iterations=4)  # 膨胀
            cnts,_ = cv2.findContours(
                mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
            frame_show = frame.copy()
            bounds = nms_cnts(cnts, mask, 400)

            for b in bounds:
                x, y, w, h = b

                cv2.rectangle(frame_show, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('video1', mask)
            cv2.imshow('video2', frame_show)
            prevframe = nextframe  # 帧差法 背景变化
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    cv2.destroyAllWindows()
    cap.release()


def catch_video(name, video_index):

    cap = cv2.VideoCapture(video_index)  # 创建摄像头识别类

    if not cap.isOpened():
        # 如果没有检测到摄像头，报错
        raise Exception('Check if the camera is on.')

    while cap.isOpened():
        catch, frame = cap.read()  # 读取每一帧图片

        cv2.imshow(name, frame)  # 在window上显示图片

        key = cv2.waitKey(10)

        if key & 0xFF == ord('q'):
            # 按q退出
            break

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break

    # 释放摄像头
    cap.release()
    cv2.destroyAllWindows()

def getFrameImgs(videoPath):
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        # 如果没有检测到摄像头，报错
        raise Exception('Check if the camera is on.')
    count = 0
    while True:
        ret, frame = cap.read()
        count += 1
        if not ret:
            break
        if ret and count % 100==0:
            cv2.imwrite("F:\\TrashImg\\"+str(count)+".jpg",frame)
            print("img "+str(count)+" is saved.")

if __name__ == '__main__':
    path = "E:\\test.mp4"
    getFrameImgs(path)
