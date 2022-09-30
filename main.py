# / -*-codeing = utf-8  -*-
# TIME : 2022/9/22 13:52
# File : main
# 计算视差图
import cv2 as cv
import numpy as np
import os


# 选取左右文件夹里的第一张照片来测试
def get_pic(train_dir2, train_dir3):
    pic_file2 = os.listdir(train_dir2)
    pic_data2 = cv.imread(os.path.join(train_dir2, pic_file2[0]))
    pic_file3 = os.listdir(train_dir3)
    pic_data3 = cv.imread(os.path.join(train_dir3, pic_file3[0]))

    return pic_data2, pic_data3


left_dir2 = r'.\SaveImage\left'
right_dir3 = r'.\SaveImage\right'


def get_f_r(pos, arr):
    l = r = pos
    for i in range(pos, 1, -1):
        if arr[i] > 0:
            l = i
            break
    for i in range(pos, len(arr) - 1):
        if arr[i] > 0:
            r = i
            break

    return l, r


if __name__ == '__main__':

    dst, dst2 = get_pic(left_dir2, right_dir3)
    # 转换为灰度图后计算视差图
    imgL = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    imgR = cv.cvtColor(dst2, cv.COLOR_BGR2GRAY)
    window_size = 9
    min_disp = 0
    num_disp = 112 - min_disp
    # 双目立体匹配算法SGBM（立体匹配）一般修改numDisparities、blockSize、uniquenessRatio这三个参数，其他参数可以按照默认的
    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                  numDisparities=num_disp,
                                  blockSize=81,
                                  P1=8 * 3 * window_size ** 2,
                                  P2=32 * 3 * window_size ** 2,
                                  disp12MaxDiff=1,
                                  uniquenessRatio=10,
                                  speckleWindowSize=100,
                                  speckleRange=32
                                  )
    print("stereo:", stereo)
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    print("disp:", disp)
    new_disp = (disp - min_disp) / num_disp
    print(new_disp.shape[0])  # 图像的高480
    print(new_disp.shape[1])  # 图像的宽640
    count = 0
    for j in range(new_disp.shape[0]):
        if new_disp[0, j] > 0:
            # 确定是第几行是 > 0的数值
            count = j
            break
    # 按照new_disp的数组宽高创建一个全0的数组，空白图像
    new_disp_pp = np.zeros_like(new_disp)
    # 遍历所有像素点
    for i in range(new_disp.shape[0]):
        for j in range(count, new_disp.shape[1] - 1):
            if new_disp[i, j] <= 0:
                left, right = get_f_r(j, new_disp[i])
                # up, down  = get_f_r(i, new_disp[:, j])
                new_disp_pp[i, j] = (new_disp[i, left - 1] + new_disp[i, right + 1]) / 2
            else:
                new_disp_pp[i, j] = new_disp[i, j]
    new_disp_pp = cv.medianBlur(new_disp_pp, 5)

    cv.imshow('org_img', dst)
    cv.imshow('org_disp', new_disp)
    cv.imshow("disp_pp", new_disp_pp)
    cv.waitKey()
    '''
    right_stereo = cv.StereoSGBM_create(minDisparity=-num_disp,
                                  numDisparities=num_disp,
                                  blockSize=8,
                                  P1=8 * 3 * window_size ** 2,
                                  P2=32 * 3 * window_size ** 2,
                                  disp12MaxDiff=1,
                                  uniquenessRatio=10,
                                  speckleWindowSize=100,
                                  speckleRange=32
                                  )

    right_disp1 = right_stereo.compute(imgR, imgL).astype(np.float32) / 16.0
    right_new_disp = -(right_disp1 - min_disp) / num_disp

    for i in range(375):
        for j in range(1242):
            if right_new_disp[i, j] >= 1:
                right_new_disp[i, j] = 0

    right_new_disp = cv.medianBlur(right_new_disp, 5)
    '''

    cv.waitKey()
    cv.destroyAllWindows()
