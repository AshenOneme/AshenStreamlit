# -*- coding: UTF-8 -*-
import plotly.graph_objs as go
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def salt_pepper(image, salt, pepper):
    """
    添加椒盐噪声的图像
    :param image: 输入图像
    :param salt: 盐比例
    :param pepper: 椒比例
    :return: 添加了椒盐噪声的图像
    """
    height = image.shape[0]
    width = image.shape[1]
    pertotal = salt + pepper    #总噪声占比
    noise_image = image.copy()
    noise_num = int(pertotal * height * width)
    for i in range(noise_num):
        rows = np.random.randint(0, height-1)
        cols = np.random.randint(0,width-1)
        if(np.random.randint(0,100)<salt*100):
            noise_image[rows][cols] = 255
        else:
            noise_image[rows][cols] = 0
    return noise_image
 
 
def low_pass_filtering(image, radius):
    """
    低通滤波函数
    :param image: 输入图像
    :param radius: 半径
    :return: 滤波结果
    """
    # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
    fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 对fft进行中心化，生成的dshift仍然是一个三维数组
    dshift = np.fft.fftshift(fft)
 
    # 得到中心像素
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)
 
    # 构建掩模，256位，两个通道
    mask = np.zeros((rows, cols, 2), np.float32)
    mask[mid_row - radius:mid_row + radius, mid_col - radius:mid_col + radius] = 1
 
    # 给傅里叶变换结果乘掩模
    fft_filtering = dshift * mask
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv2.idft(ishift)
    image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
    # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
    cv2.normalize(image_filtering, image_filtering, 0.01, 1, cv2.NORM_MINMAX)
    return image_filtering
 
 
def high_pass_filtering(image, radius, n):
    """
    高通滤波函数
    :param image: 输入图像
    :param radius: 半径
    :param n: ButterWorth滤波器阶数
    :return: 滤波结果
    """
    # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
    fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 对fft进行中心化，生成的dshift仍然是一个三维数组
    dshift = np.fft.fftshift(fft)
 
    # 得到中心像素
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)
 
    # 构建ButterWorth高通滤波掩模
 
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(0, rows):
        for j in range(0, cols):
            # 计算(i, j)到中心点的距离
            d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
            try:
                mask[i, j, 0] = mask[i, j, 1] = 1 / (1 + pow(radius / d, 2*n))
            except ZeroDivisionError:
                mask[i, j, 0] = mask[i, j, 1] = 0
    # 给傅里叶变换结果乘掩模
    fft_filtering = dshift * mask
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv2.idft(ishift)
    image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
    # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
    cv2.normalize(image_filtering, image_filtering, 0.01, 1, cv2.NORM_MINMAX)
    return image_filtering
 
 
def bandpass_filter(image, radius, w, n=1):
    """
    带通滤波函数
    :param image: 输入图像
    :param radius: 带中心到频率平面原点的距离
    :param w: 带宽
    :param n: 阶数
    :return: 滤波结果
    """
    # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
    fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 对fft进行中心化，生成的dshift仍然是一个三维数组
    dshift = np.fft.fftshift(fft)
 
    # 得到中心像素
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)
 
    # 构建掩模，256位，两个通道
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(0, rows):
        for j in range(0, cols):
            # 计算(i, j)到中心点的距离
            d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
            if radius - w / 2 < d < radius + w / 2:
                mask[i, j, 0] = mask[i, j, 1] = 1
            else:
                mask[i, j, 0] = mask[i, j, 1] = 0
 
    # 给傅里叶变换结果乘掩模
    fft_filtering = dshift * np.float32(mask)
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv2.idft(ishift)
    image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
    # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
    cv2.normalize(image_filtering, image_filtering, 0.01, 1, cv2.NORM_MINMAX)
    return image_filtering
 
 
def bandstop_filter(image, radius, w, n=1):
    """
    带通滤波函数
    :param image: 输入图像
    :param radius: 带中心到频率平面原点的距离
    :param w: 带宽
    :param n: 阶数
    :return: 滤波结果
    """
    # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
    fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 对fft进行中心化，生成的dshift仍然是一个三维数组
    dshift = np.fft.fftshift(fft)
 
    # 得到中心像素
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)
 
    # 构建掩模，256位，两个通道
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(0, rows):
        for j in range(0, cols):
            # 计算(i, j)到中心点的距离
            d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
            if radius - w / 2 < d < radius + w / 2:
                mask[i, j, 0] = mask[i, j, 1] = 0
            else:
                mask[i, j, 0] = mask[i, j, 1] = 1
 
    # 给傅里叶变换结果乘掩模
    fft_filtering = dshift * np.float32(mask)
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv2.idft(ishift)
    image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
    # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
    cv2.normalize(image_filtering, image_filtering, 0.01, 1, cv2.NORM_MINMAX)
    return image_filtering

st.set_page_config(page_title="Image Filtering", page_icon="⚗")
st.markdown("# Image Filtering")
st.sidebar.header("Image Filtering")
uploaded_file = st.file_uploader("Please upload a picture")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.title("Here is the original image!")
    st.image(opencv_image, channels="BGR")
    
    opencv_image = cv2.imdecode(file_bytes, 0)
    
    st.title("Salt pepper image!")
    salt=st.slider('salt',value=float(0.01),min_value=float(0),max_value=float(1),step=0.01)
    pepper=st.slider('pepper',value=float(0.01),min_value=float(0),max_value=float(1),step=0.01)
    image_salt_pepper = salt_pepper(opencv_image, salt,pepper)
    st.image(image_salt_pepper)
    
    st.title("Low pass filtering image!")
    Initial1=st.slider('Low Pass Radius',value=int(1),min_value=int(1),max_value=100,step=1)
    image_low_pass_filtering = low_pass_filtering(opencv_image, Initial1)
    st.image(image_low_pass_filtering)
    
    st.title("High pass filtering image!")
    Initial2=st.slider('High Pass Radius',value=int(1),min_value=int(1),max_value=100,step=1)
    ButterWorth=st.number_input('ButterWorth:',value=int(1))
    image_high_pass_filtering = high_pass_filtering(opencv_image, Initial2,ButterWorth)
    st.image(image_high_pass_filtering)
    
    st.title("Bandstop filtering image!")
    Initial3=st.slider('Bandstop Radius',value=int(1),min_value=int(1),max_value=100,step=1)
    Initial4=st.slider('Bandstop W',value=int(1),min_value=int(1),max_value=100,step=1)
    BandstopN=st.number_input('Bandstop N:',value=int(1))
    image_bandstop_filtering = bandstop_filter(opencv_image, Initial3,Initial4,BandstopN)
    st.image(image_bandstop_filtering)
    
    #cv2.imwrite('test.jpg',opencv_image)
    
    
