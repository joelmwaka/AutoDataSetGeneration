import cv2
import copy
import json
import numpy as np


class FeaturePicker():

    def __init__(self, img_input, num_features, path_json=None):

        self.input_image = img_input
        self.input_image_copy = copy.deepcopy(self.input_image)
        self.num_features = num_features
        self.features = []
        self.path_json = path_json
        self.print_done = False

        assert self.num_features != 0, "num_features parameter cannot be 0"


    def _click_event(self, event, x, y, flags, params):

        # Left mouse click - Select Point
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.input_image_copy, (x, y), 3, (0, 0, 255), -1)
            self.features.append([x, y])
            self.print_done = False
        
        # Right mouse click - Restart Selection
        if event == cv2.EVENT_RBUTTONDOWN:
            self.input_image_copy = copy.deepcopy(self.input_image)
            self.features.clear()
            self.print_done = False


    def _event_loop(self):
        
        cv2.namedWindow('Feature Picker')
        cv2.setMouseCallback('Feature Picker', self._click_event)
        
        while True:

            if not self.print_done:
                if len(self.features) < self.num_features:
                    print("INSTRUCTION: Pick Point for Feature Index " + str(len(self.features)))
                else:
                    print("INSTRUCTION: Press 'ESC' to end or Right Click to Restart.")
                self.print_done = True

            cv2.imshow("Feature Picker", self.input_image_copy)
            k = cv2.waitKey(1) & 0xFF

            if k == 27: # If 'ESC' is pressed.
                if len(self.features) == self.num_features:
                    break
                elif len(self.features) < self.num_features:
                    print("NOTE: Number of features selected is less than number of required features.")
                else:
                    print("NOTE: Too many features selected. Right click on image to restart selection.")
        
        cv2.destroyAllWindows()


    def _create_json(self):

        dict_features = {}

        for index, feature in enumerate(self.features):
            dict_features[str(index)] = feature

        with open(self.path_json + "features.json", "w") as outfile:
            json.dump(dict_features, outfile)


    def run(self):

        print("================")
        print("Feature Selector")
        print("================")

        self._event_loop()
        if self.path_json is not None:
            self._create_json()

