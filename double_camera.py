# / -*-codeing = utf-8  -*-
# TIME : 2022/9/22 11:30
# File : double_camera  双目摄像头拍照镜头单独拍照
import cv2
import time

AUTO = False  # 自动拍照，或手动按s键拍照
INTERVAL = 2  # 自动拍照间隔

cv2.namedWindow("left")
cv2.namedWindow("right")
camera = cv2.VideoCapture(0)
# 设置分辨率 左右摄像机同一频率，同一设备ID；左右摄像机总分辨率1280x480；分割为两个640x480、640x480
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
utc = time.time()
folder_l = "./SaveImage/left/"  # 拍照文件目录
folder_r = "./SaveImage/right/"  # 拍照文件目录


def shot(pos, frame, dir):
    global counter
    if dir:
        path = folder_l + pos + "_" + str(counter) + ".jpg"
    else:
        path = folder_r + pos + "_" + str(counter) + ".jpg"
    cv2.imwrite(path, frame)
    print("snapshot saved into: " + path)


while True:
    ret, frame = camera.read()
    if ret:
        # 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
        left_frame = frame[:, 0:640, :]
        right_frame = frame[:, 640:1280, :]
        cv2.imshow("left", left_frame)
        cv2.imshow("right", right_frame)

        now = time.time()
        left = 1
        right = 0
        if AUTO and now - utc >= INTERVAL:
            shot("left", left_frame, 1)
            shot("right", right_frame, 0)
            counter += 1
            utc = now

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            shot("left", left_frame, 1)
            print("left_frame: ", left_frame)
            shot("right", right_frame, 0)
            counter += 1
    else:
        print("not found camera")
camera.release()
cv2.destroyWindow("left")
cv2.destroyWindow("right")