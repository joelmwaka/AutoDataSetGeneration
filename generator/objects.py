import os
import re
import cv2
import random
import logging
import numpy as np


class Object:

    def __init__(self, path_source_image, path_labels, file_name):

        self.path_image = path_source_image
        self.path_labels = path_labels
        self.file_name = file_name
        self.image = None

        self.label = ""
        self.bounding_box = None
        self.num_features = None
        self.features = None

    def annotate(self, skip, features):

        path_annotation = os.path.join(self.path_labels, self.file_name.split(".")[0] + ".txt")
        annotate = None

        if skip:
            if not os.path.exists(path_annotation):  # skip, but no label file
                logging.warning(f"No label file exists for object {self.file_name}.")
                user_input = input(f"Prompt: No label file exists for object {self.file_name}. "
                                   f"Annotate? (y/n): ")

                while True:
                    if user_input.lower() == "y":
                        annotate = True
                        break
                    elif user_input.lower() == "n":
                        annotate = False
                        break
                    else:
                        user_input = input(f"Prompt: Unexpected input '{user_input}'. "
                                           f"Try again: ")

            else:  # skip and label file exists
                annotate = False

        else:  # not skip
            if os.path.exists(path_annotation):  # label file exists

                user_input = input("Prompt: An annotation file for this object exists. "
                                   "Overwrite? (y/n): ")

                while True:
                    if user_input.lower() == "y":
                        annotate = True
                        break
                    elif user_input.lower() == "n":
                        annotate = False
                        break
                    else:
                        user_input = input(f"Prompt: Unexpected input '{user_input}'. "
                                           f"Try again: ")

            else:  # not skip and file doesn't exist
                annotate = True

        if annotate:
            self.image = cv2.imread(self.path_image)
            cv2.namedWindow("Object")

            print(f"Object Image: {self.path_image}")
            while True:
                cv2.imshow("Object", self.image)
                key = cv2.waitKey(0)
                if key == 13:  # "Enter" key pressed
                    break

            cv2.destroyWindow("Object")

            # label objects
            pattern = r'^[a-zA-Z_]+$'
            while not self.label:
                user_input = input("Prompt: Enter label: ")

                if not user_input:
                    print("Info: Object label can not be empty!")
                elif not bool(re.match(pattern, user_input)):
                    print("Info: Object label may contain upper- and lowercase letters and underscores!")
                else:
                    self.label = user_input

            # features
            if features:
                while self.num_features is None:
                    user_input = input("Prompt: Enter number of features: ")

                    if not user_input.isdigit():
                        print(f"Info: {user_input} is not a number.")
                    else:
                        self.num_features = int(user_input)

                # annotate features
                if self.num_features != 0:
                    feature_picker = FeaturePicker(image=self.image, num_features=self.num_features)
                    self.features = feature_picker.eventloop()
                else:
                    self.features = np.array([])

            # get bounding box
            height, width, _ = self.image.shape
            self.bounding_box = np.array([0, 0, height, width], dtype=float)

            # save annotation
            with open(path_annotation, "w") as file:
                file.write(f"path_to_image: {self.path_image}\n")
                file.write(f"label: {self.label}\n")
                file.write(f"bounding_box: {self.bounding_box.tolist()}\n")
                if features:
                    file.write(f"num_features: {self.num_features}\n")
                    file.write(f"features: {self.features.tolist()}")

        else:
            self.image = cv2.imread(self.path_image)
            with open(path_annotation, "r") as file:
                for line in file:
                    data = line.split(": ")
                    key = data[0]
                    value = data[1]

                    if key == "label":
                        self.label = value[:-1]
                    elif key == "bounding_box":
                        bbox = []
                        vals = value[1:-2].split(",")
                        for val in vals:
                            bbox.append(float(val.replace(" ", "")))
                        self.bounding_box = np.array(bbox, dtype=float)
                    elif key == "num_features":
                        self.num_features = int(value)
                    elif key == "features":
                        features = []
                        feats = value.replace("[", "").replace("]", "").replace(" ", "").split(",")
                        for i in range(self.num_features):
                            feat = [float(feats[i*2]), float(feats[i*2 + 1])]
                            features.append(feat)
                        self.features = np.array(features, dtype=float)
                    else:
                        if key != "path_to_image":
                            print(f"Info: Unexpected key '{key}'.")


class FeaturePicker:

    def __init__(self, image: np.ndarray, num_features: int):

        self.input_image_copy = np.copy(image)
        self.num_features = num_features
        self.features = []
        self.print_done = False

    def _click_event(self, event, x, y, flags, params):

        # Left mouse click - Select Point
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.input_image_copy, (x, y), 3, (0, 0, 255), -1)
            self.features.append([x, y])
            self.print_done = False

        # Right mouse click - Restart Selection
        if event == cv2.EVENT_RBUTTONDOWN:
            self.input_image_copy = np.copy(self.input_image_copy)
            self.features.clear()
            self.print_done = False

    def eventloop(self):

        self.features.clear()
        cv2.namedWindow("Feature Picker")
        cv2.setMouseCallback("Feature Picker", self._click_event)

        while True:

            cv2.imshow("Feature Picker", self.input_image_copy)

            if not self.print_done:
                if len(self.features) < self.num_features:
                    print(f"Instruction: Pick point for feature {len(self.features)}.")
                else:
                    print("Instruction: Press 'ESC' to end or right click to restart.")
                self.print_done = True

            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # if 'ESC' is pressed.
                if len(self.features) == self.num_features:
                    break
                elif len(self.features) < self.num_features:
                    print("Note: Number of features selected is less than number of required features.")
                else:
                    print("Note: Too many features selected. Right click on image to restart selection.")

        cv2.destroyWindow("Feature Picker")

        return np.array(self.features, dtype=float)


class Objects:

    def __init__(self, settings: dict):

        logging.debug("Initializing the objects class.")

        self.settings = settings
        self.skip = self.settings["skip"]
        self.features = self.settings["features"]
        self.path_images = self.settings["path_images"]
        self.path_labels = self.settings["path_labels"]
        self.list_objects = []
        self.eventloop()

        logging.debug("Done initializing the objects class.")

    def eventloop(self):

        file_names = os.listdir(self.path_images)

        for fname in file_names:
            obj = Object(
                path_source_image=os.path.join(self.path_images, fname),
                path_labels=self.path_labels, file_name=fname
            )
            obj.annotate(skip=self.skip, features=self.features)
            self.list_objects.append(obj)

    def get_object(self) -> Object:

        if not self.list_objects:
            raise ValueError("Objects were not initialized.")

        return random.choice(self.list_objects)
