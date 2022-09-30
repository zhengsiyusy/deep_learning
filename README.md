## 计算图像深度信息，使用SGBM算法

相机使用的是单usb的双目相机，基线距离为6cm,本项目主要使用的文件是前2个

Camera_calibrate.py		计算图像深度信息主函数

matlab_calibrateCamera.py    matlab获取相机的内参、旋转矩阵、平移矩阵、畸变等参数配置信息

double_camera.py    双目相机拍照使用，为相机标定参数提供照片

find_corners_demo.py    相机标定查找角点的例子

SGBM_demo.py  / shichatu_cal_demo    SGBM算法简单测试的例子（不是很准确）

