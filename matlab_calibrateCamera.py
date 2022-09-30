# / -*-codeing = utf-8  -*-
# TIME : 2022/9/28 17:30
# File : matlab_calibrateCamera  使用matlab标定获取的参数，进行应用

import cv2
import numpy as np

# 左相机内参stereoParams.CameraParameters1.Intrinsics
left_camera_matrix = np.array([[4.830476884573363e+02, 0., 3.435350849801462e+02],
                               [0., 4.821550044790280e+02, 2.679967872197112e+02],
                               [0., 0., 1.]])

# 右相机内参stereoParams.CameraParameters2.Intrinsics
right_camera_matrix = np.array([[4.839786993692161e+02, 0., 3.438247964707048e+02],
                                [0., 4.825644697978253e+02, 2.687565870132457e+02],
                                [0., 0., 1.]])
# 左相机畸变值
left_distortion = np.array([[0.036731469027784, 0.132806920066971, -9.026014028421839e-04, 0.005720312915826, 0]])
# 右相机畸变值
right_distortion = np.array([[0.063443929616310, -0.029519746818953, -0.002919439293187, 0.003925355279831, 0]])

# 旋转矩阵
R = np.array([[0.999950666276819, -1.829223329498642e-04, 0.009931341901582],
              [1.655271177239093e-04, 0.999998450949637, 0.001752341034224],
              [-0.009931647059744, -0.001750610678316, 0.999949147581483]])
# 平移矩阵
T = np.array([-59.605889023614150, -0.049970151922868, -0.043241899652839])
size = (640, 480)  # 图像尺寸
