import cv2


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
        if ret and count==7:
            count=0
            diff = cv2.absdiff(prevframe, nextframe)
            cv2.imshow('video1', diff)
            cv2.imshow('video2', frame)
            prevframe = nextframe  # 帧差法 背景变化
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    cv2.destroyAllWindows()
    cap.release()


def catch_video(name, video_index):
    # cv2.namedWindow(name)

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

if __name__ == '__main__':
    path = "G:\\2999_巴黎婚纱摄影门前_1_1602822556_952D7CCE\\2999_巴黎婚纱摄影门前_1_1FA99388_1602822556_1.mp4"
    getFrameDiff(path)
