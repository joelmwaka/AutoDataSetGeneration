import argparse
import cv2
import numpy as np
from utils.my_util_feature_picker import FeaturePicker
from utils.my_util_boundbox import BoundBoxPicker
from utils.my_util_image_operations import *


def main():

    '''
    Step 0: Preparation
    '''
    # TODO: Parsing Arguments

    # parser = argparse.ArgumentParser(description='Automatic H-Tafel Data Set Generation.')
    # parser.add_argument("--steps", type=str, default="1")
    # args = parser.parse_args()
    # steps = args.steps

    run_steps = {'step1': False, 'step2': True, 'step3': False}

    num_features_img_object = 12
    path_OriginalObjectImages = './Data/OriginalObjectImages/'
    path_RandomUnprocessedImages = './Data/RandomUnprocessedImages/'
    path_RandomProcessedImages = './Data/RandomProcessedImages/'
    path_GeneratedAnnotatedDataset = './Data/GeneratedAnnotatedData/'

    file_name_img_orig_object = 'original_image_h_tafel.jpg'
    img_orig_object = cv2.imread(path_OriginalObjectImages + file_name_img_orig_object)


    '''
    Step 1: Manually select key points from original object image
    '''

    if run_steps['step1']:

        feature_picker = FeaturePicker(img_orig_object, num_features=num_features_img_object, path_json=path_OriginalObjectImages)
        feature_picker.run()

        boundbox_picker = BoundBoxPicker(img_orig_object, path_json=path_OriginalObjectImages)
        boundbox_picker.run()


    '''
    Step 2: Create Data Set
    '''

    if run_steps['step2']:

        # Alter 
        img_random = np.random.randint(255, size=(320,480,3),dtype=np.uint8)
        cv2.imshow('RGB', img_random)
        cv2.waitKey(0)

        img_object_noisy = img_add_gaussian_noise(img_orig_object)
        cv2.imshow("Noisy Image", img_object_noisy)
        cv2.waitKey(0)



if __name__ == "__main__":
    main()
