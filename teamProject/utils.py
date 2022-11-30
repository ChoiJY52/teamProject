import numpy as np, cv2

def equalization(image):
    image_Ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(image_Ycrcb)
    Y = cv2.equalizeHist(Y)
    dst_Ycrcb = cv2.merge((Y,Cr,Cb))
    dst = cv2.cvtColor(dst_Ycrcb, cv2.COLOR_YCrCb2BGR)
    return dst

def search_value_idx(hist, bias = 0):
    for i in range(hist.shape[0]):
        idx = np.abs(bias - i)                     # 검색 위치 (처음 또는 마지막)
        if hist[idx] > 0:  return idx                             # 위치 반환
    return -1
def stretching(image):
    bsize, ranges = [64], [0, 256]  # 계급 개수 및 화소 범위
    hist = cv2.calcHist([image], [0], None, bsize, ranges)

    bin_width = ranges[1] / bsize[0]  # 계급 너비
    high = search_value_idx(hist, bsize[0] - 1) * bin_width
    low = search_value_idx(hist, 0) * bin_width

    idx = np.arange(0, 256)
    idx = (idx - low) * 255 / (high - low)  # 수식 적용하여 인덱스 생성
    idx[0:int(low)] = 0
    idx[int(high + 1):] = 255

    dst = cv2.LUT(image, idx.astype('uint8'))
    return dst

def brightness_control(image, flags):
    B, G, R = cv2.split(image)
    if flags == 0:
        B += 80; G += 80; R += 80
    elif flags == 1:
        B -= 80; G -= 80; R -= 80
    dst = cv2.merge((B,G,R))
    return dst