# / -*-codeing = utf-8  -*-
# TIME : 2022/9/23 8:24
# File : test_cam
# import cv2
# import time
#
# AUTO = False  # 自动拍照，或手动按s键拍照
# INTERVAL = 2  # 自动拍照间隔
#
# cv2.namedWindow("left")
# cv2.namedWindow("right")
#
# camera = cv2.VideoCapture(0)
# # 设置分辨率 左右摄像机同一频率，同一设备ID；左右摄像机总分辨率1280x480；分割为两个640x480、640x480
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
# # camera1 = cv2.VideoCapture(1)
# # camera1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# # camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# counter = 0
# utc = time.time()
# folder = "./SaveImage/"  # 拍照文件目录
# cnt = 1
#
#
# def shot(pos, frame):
#     global counter
#     path = folder + pos + "_" + str(counter) + ".jpg"
#
#     cv2.imwrite(path, frame)
#     print("snapshot saved into: " + path)
#
#
# while True:
#     camera1 = cv2.VideoCapture(1400)
#     camera1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
#     ret, frame = camera.read()
#     ret1, frame1 = camera1.read()
#     if ret:
#         # 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
#         cv2.imshow("left", frame)
#         # now = time.time()
#         # if AUTO and now - utc >= INTERVAL:
#         #     shot("left", frame)
#         #     counter += 1
#         #     utc = now
#         #
#         # key = cv2.waitKey(1)
#         # if key == ord("q"):
#         #     break
#         # elif key == ord("s"):
#         #     shot("left", frame)
#         #     counter += 1
#     else:
#         print("camera not found")
#     if ret1:
#         # 裁剪坐标为[y0:y1, x0:x1] HEIGHT*WIDTH
#         # print("cnt", cnt)
#         cv2.imshow("right", frame1)
#
#         # now1 = time.time()
#         # if AUTO and now1 - utc >= INTERVAL:
#         #     shot("right", frame1)
#         #     counter += 1
#         #     utc = now1
#         #
#         # key1 = cv2.waitKey(1)
#         # if key1 == ord("q"):
#         #     break
#         # elif key1 == ord("s"):
#         #     shot("right", frame1)
#         #     counter += 1
#     else:
#         # cnt = cnt+1
#         print("not found camera")
#
# camera.release()
# camera1.release()
# cv2.destroyWindow("left")
# cv2.destroyWindow("right")


import cv2

capture = cv2.VideoCapture(0)
capture_usb = cv2.VideoCapture(1400)
# 打开自带的摄像头
if capture.isOpened():
    if capture_usb.isOpened():
        # 以下设置显示屏的宽高
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture_usb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture_usb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 持续读取摄像头数据
while True:
    read_code, frame = capture.read()
    read_code2, frame2 = capture_usb.read()
    if not read_code or not read_code2:
        #
        print('error')
    else:
        cv2.imshow("screen_title", frame)
        cv2.imshow("screen_title_usb", frame2)
    # 输入 q 键，保存当前画面为图片
    if cv2.waitKey(1) == ord('q'):
        # 设置图片分辨率
        frame = cv2.resize(frame, (1920, 1080))
        cv2.imwrite('pic.jpg', frame)
        capture_usb.release()
        break
# 释放资源
capture.release()
cv2.destroyWindow("screen_title")