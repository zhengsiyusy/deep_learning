# / -*-codeing = utf-8  -*-
# TIME : 2022/9/23 14:24
# File : test_bd  单目相机标定opencv
import cv2
import math
import os
import numpy as np
import matlab_calibrateCamera as camera_configs

window_size = 20
min_disp = 16
num_disp = 192 - min_disp
blockSize = window_size
uniquenessRatio = 1
speckleRange = 3
speckleWindowSize = 3
disp12MaxDiff = 200
SGBM_P1 = 600
SGBM_P2 = 2400


# 添加点击事件，打印当前点的距离    鼠标回调函数
def callbackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        print('\n像素坐标 x = %d, y = %d：' % (x, y))
        print("世界坐标xyz 是： ", threeD[y][x][0] / 1000.0, threeD[y][x][1] / 1000.0, threeD[y][x][2] / 1000.0, "m")

        distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2)
        print("毫米距离： ", distance)
        distance = distance / 1000.0  # mm ->m(毫米转米)
        print("距离是： ", distance, "m")


def set_Window_Trackbar():
    cv2.namedWindow("left")
    cv2.namedWindow("right")
    cv2.namedWindow("depth")
    cv2.moveWindow("left", 0, 0)
    cv2.moveWindow("right", 600, 0)
    cv2.createTrackbar("num", "depth", 0, 10, lambda x: None)
    cv2.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)


# def calib_camera(calib_dir, pattern_size=(8, 5), draw_points=True):
#     """
#     calibrate camera
#     :param calib_dir: str
#     :param pattern_size: (x, y), the number of points in x, y axes in the chessboard
#     :param draw_points: bool, whether to draw the chessboard points
#     """
#     # store 3d object points and 2d image points from all the images
#     global img_gray
#     object_points = []
#     image_points = []
#
#     # 3d object point coordinate
#     xl = np.linspace(0, pattern_size[0], pattern_size[0], endpoint=False)
#     yl = np.linspace(0, pattern_size[1], pattern_size[1], endpoint=False)
#     xv, yv = np.meshgrid(xl, yl)
#     object_point = np.insert(np.stack([xv, yv], axis=-1), 2, 0, axis=-1).astype(np.float32).reshape([-1, 3])
#
#     # set termination criteria
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#     # load image
#     img_dir = calib_dir
#     assert os.path.isdir(img_dir), 'Path {} is not a dir'.format(img_dir)
#     imagenames = os.listdir(img_dir)
#     for imagename in imagenames:
#         if not os.path.splitext(imagename)[-1] in ['.jpg', '.png', '.bmp', '.tiff', '.jpeg']:
#             continue
#         img_path = os.path.join(img_dir, imagename)
#         img = cv2.imread(img_path)
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # find chessboard points
#         ret, corners = cv2.findChessboardCorners(img_gray, patternSize=pattern_size)
#         if ret:
#             # add the corresponding 3d points to the summary list
#             object_points.append(object_point)
#             # if chessboard points are found, refine them to SubPix level (pixel location in float)
#             corners_refined = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
#             # add the 2d chessboard points to the summary list
#             image_points.append(corners.reshape([-1, 2]))
#             # visualize the points
#             if draw_points:
#                 cv2.drawChessboardCorners(img, pattern_size, corners_refined, ret)
#                 if img.shape[0] * img.shape[1] > 1e6:
#                     scale = round((1. / (img.shape[0] * img.shape[1] // 1e6)) ** 0.5, 3)
#                     img_draw = cv2.resize(img, (0, 0), fx=scale, fy=scale)
#                 else:
#                     img_draw = img
#
#                 cv2.imshow('img', img_draw)
#                 cv2.waitKey(0)
#
#     assert len(image_points) > 0, 'Cannot find any chessboard points, maybe incorrect pattern_size has been set'
#     # calibrate the camera, note that ret is the rmse of reprojection error, ret=1 means 1 pixel error
#     reproj_err, k_cam, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points,
#                                                                        image_points,
#                                                                        img_gray.shape[::-1],
#                                                                        None,
#                                                                        None,
#                                                                        criteria=criteria)
#     print("ret:", reproj_err)
#     print("mtx:\n", k_cam)  # 内参数矩阵
#     print("dist畸变值:\n", dist_coeffs)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
#     # print("rvecs旋转（向量）外参:\n", rvecs)  # 旋转向量  # 外参数
#     # print("tvecs平移（向量）外参:\n", tvecs)  # 平移向量  # 外参数
#
#     return k_cam, dist_coeffs, rvecs, tvecs


# 单目相机畸变矫正（使用）
# def cam_aberration_rectify(cam_frame, camera_matrix, distortion):
#     h1, w1 = cam_frame.shape[:2]
#     # 获取最佳的相机矩阵参数
#     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, (h1, w1), 0, (h1, w1))
#     # 纠正畸变
#     dst1 = cv2.undistort(cam_frame, camera_matrix, distortion, None, newcameramtx)
#     # dst2 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
#     mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, newcameramtx, (w1, h1), 5)
#     dst2 = cv2.remap(cam_frame, mapx, mapy, cv2.INTER_LINEAR)
#     # 裁剪图像，输出纠正畸变以后的图片
#     x, y, w1, h1 = roi
#     dst1 = dst1[y:y + h1, x:x + w1]
#
#     # cv2.imshow('frame',dst2)
#     # cv2.imshow('dst1',dst1)
#     cv2.imshow('dst1', dst1)
#     # if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q保存一张图片
#     #     cv2.imwrite("./result/frame.jpg", dst1)
#     return dst1


if __name__ == '__main__':
    set_Window_Trackbar()
    # 打开摄像机
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        (grabbed, frame) = camera.read()
        if not grabbed:
            break
        left_frame = frame[:, 0:640, :]
        right_frame = frame[:, 640:1280, :]

        h1, w1 = left_frame.shape[:2]
        h2, w2 = right_frame.shape[:2]
        if h1 == h2 and w1 == w2:
            size = (w1, h1)
        else:
            size = (640, 480)

        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(camera_configs.left_camera_matrix,
                                                                          camera_configs.left_distortion,
                                                                          camera_configs.right_camera_matrix,
                                                                          camera_configs.right_distortion, size,
                                                                          camera_configs.R, camera_configs.T)
        # 计算更正map
        left_map1, left_map2 = cv2.initUndistortRectifyMap(camera_configs.left_camera_matrix,
                                                           camera_configs.left_distortion,
                                                           R1, P1, size, cv2.CV_16SC2)
        right_map1, right_map2 = cv2.initUndistortRectifyMap(camera_configs.right_camera_matrix,
                                                             camera_configs.right_distortion,
                                                             R2, P2, size, cv2.CV_16SC2)

        # 根据更正map对图片进行重构
        img1_rectified = cv2.remap(left_frame, left_map1, left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(right_frame, right_map1, right_map2, cv2.INTER_LINEAR)
        # 将图片置为灰度图，为StereoBM作准备
        imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

        # 两个trackbar用来调节不同的参数查看效果
        num = cv2.getTrackbarPos("num", "depth")
        blockSize = cv2.getTrackbarPos("blockSize", "depth")
        if blockSize % 2 == 0:
            blockSize += 1
        if blockSize < 5:
            blockSize = 5

        # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试   /视差图
        # stereo = cv2.StereoBM_create(numDisparities=16 * num, blockSize=blockSize)
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,  # 表示可能的最小视差值。通常为0，但有时校正算法会移动图像，所以参数值也要相应调整
            numDisparities=num_disp,  # 表示最大的视差值与最小的视差值之差，这个差值总是大于0。在当前的实现中，这个值必须要能被16整除
            blockSize=window_size,
            uniquenessRatio=uniquenessRatio,  # 表示由代价函数计算得到的最好（最小）结果值比第二好的值小多少（用百分比表示）才被认为是正确的。通常在5-15之间。
            speckleRange=speckleRange,  # 指每个已连接部分的最大视差变化，如果进行斑点过滤，则该参数取正值，函数会自动乘以16、一般情况下取1或2就足够了。
            speckleWindowSize=speckleWindowSize,  # 表示平滑视差区域的最大窗口尺寸，以考虑噪声斑点或无效性。将它设为0就不会进行斑点过滤，否则应取50-200之间的某个值。
            disp12MaxDiff=disp12MaxDiff,  # 表示在左右视图检查中最大允许的偏差（整数像素单位）。设为非正值将不做检查。
            P1=SGBM_P1,  # 控制视差图平滑度的第一个参数
            P2=SGBM_P2  # 控制视差图平滑度的第二个参数，值越大，视差图越平滑。P1是邻近像素间视差值变化为1时的惩罚值，
            # p2是邻近像素间视差值变化大于1时的惩罚值。算法要求P2>P1,stereo_match.cpp样例中给出一些p1和p2的合理取值。
        )
        # 计算视差
        disparity = stereo.compute(imgL, imgR)
        # 归一化函数算法，生成深度图(灰度图)
        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 生成深度图(颜色图)
        dis_color = disparity
        dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dis_color = cv2.applyColorMap(dis_color, 2)
        # 将图片扩展至3d空间中，其z方向的值则为当前的距离
        threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., Q)
        # 鼠标点击事件
        cv2.setMouseCallback("depth", callbackFunc, threeD)

        cv2.imshow("dis_color", dis_color)
        cv2.imshow("left", img1_rectified)
        cv2.imshow("right", img2_rectified)
        cv2.imshow("depth", disp)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("./snapshot/SGBM_left.jpg", imgL)
            cv2.imwrite("./snapshot/SGBM_right.jpg", imgR)
            cv2.imwrite("./snapshot/SGBM_depth.jpg", disp)

    camera.release()
    cv2.destroyAllWindows()
