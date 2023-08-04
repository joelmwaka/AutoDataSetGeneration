import cv2
import numpy as np



def img_resize(img_input_rgb, output_size):



    pass


def img_add_gaussian_noise(img_input_rgb):

    row, col, _ = img_input_rgb.shape

    mean = 0
    var = 0.1
    sigma = var ** 0.5
    
    gaussian = np.random.random((row, col, 1)).astype(np.float32)
    gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
    img_output_rgb = cv2.addWeighted(img_input_rgb, 0.75, 0.25 * gaussian, 0.25, 0)

    return img_output_rgb