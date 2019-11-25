import sys

import numpy as np
import cv2

def floyd(img_gray):
    #black pixcel
    min_pix = 0
    #white pixcel
    max_pix = 255
    #threshold
    threshold_pix = 255 // 2

    img_size = img_gray.shape

    img_floyd = np.zeros(img_size, dtype=np.float16)
    x_1 = 7 / 16
    x_2 = 3 / 16
    x_3 = 5 / 16
    x_4 = 1/ 16

    for h in range(img_size[0]):
        for w in range(img_size[1]):
            #show white pixcel
            if(img_gray[h][w] + img_floyd[h][w]) > threshold_pix:
                pix_err = img_gray[h][w] + img_floyd[h][w] - max_pix
                img_floyd[h][w] = max_pix
            #show black pixcel
            else:
                pix_err = img_gray[h][w] + img_floyd[h][w]
                img_floyd[h][w] = min_pix
            #error propagation
            if h == (img_size[0] - 1):
                if w < (img_size[1] - 1):
                    img_floyd[h][w + 1] += (x_1 * pix_err)
            elif w == (img_size[1] - 1):
                img_floyd[h + 1][w - 1] += (x_2 * pix_err)
                img_floyd[h + 1][w] += (x_3 * pix_err)
            else:
                img_floyd[h][w + 1] += (x_1 * pix_err)
                img_floyd[h + 1][w - 1] += (x_2 * pix_err)
                img_floyd[h + 1][w] += (x_3 * pix_err)
                img_floyd[h + 1][w + 1] += (x_4 * pix_err)
    return img_floyd.astype(np.uint8)

def main():
    fname = sys.argv[1]

    #load gray scale
    img_gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    img_bin = floyd(img_gray)
    cv2.imwrite('out.png', img_bin)


if __name__ == '__main__':
    main()
