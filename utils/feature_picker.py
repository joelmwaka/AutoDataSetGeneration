import os
import cv2
import copy
import json


class SingleImageFeaturePicker:

    def __init__(self):

        self.input_image = None
        self.input_image_copy = None
        self.num_features = None
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
            self.input_image_copy = copy.deepcopy(self.input_image)
            self.features.clear()
            self.print_done = False

    def eventloop(self, path_to_image):

        img_input = cv2.imread(path_to_image)
        self.input_image = copy.deepcopy(img_input)
        self.input_image_copy = copy.deepcopy(img_input)
        self.num_features = None
        self.features.clear()
        cv2.namedWindow('Feature Picker')
        cv2.setMouseCallback('Feature Picker', self._click_event)

        while True:

            cv2.imshow("Feature Picker", self.input_image_copy)

            # set number of features
            if self.num_features == None:
                while True:
                    user_input = input("Enter the number of features you wish to select in the image: ")
                    self.num_features = int(user_input)

                    if self.num_features > 0:
                        print(f"Info: Please pick {self.num_features} features.")
                        break
                    else:
                        print(f"Info: Expected integer above 0, got {self.num_features}. Try again.")

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

        cv2.destroyAllWindows()

        return self.features


class MultipleImageFeaturePicker:

    def __init__(self, path_to_objects):

        self.path_to_objects = path_to_objects
        self.path_to_annotations = os.path.join(path_to_objects, "annotations")
        self.feature_picker = SingleImageFeaturePicker()
        self.all_features = {}

    def _create_json(self, filename, features):
        """Save features for a single image to a JSON file."""
        # Create annotations directory if it doesn't exist
        os.makedirs(self.path_to_annotations, exist_ok=True)

        # Convert features list to dict format
        dict_features = {}
        for index, feature in enumerate(features):
            dict_features[str(index)] = feature

        # Create JSON filename based on image filename
        base_name = os.path.splitext(filename)[0]
        json_path = os.path.join(self.path_to_annotations, f"{base_name}_features.json")

        with open(json_path, "w") as outfile:
            json.dump(dict_features, outfile, indent=2)

        print(f"Info: Saved features to '{json_path}'.")

    def _save_all_features(self):
        """Save all features to a combined JSON file."""
        os.makedirs(self.path_to_annotations, exist_ok=True)

        json_path = os.path.join(self.path_to_annotations, "all_features.json")
        with open(json_path, "w") as outfile:
            json.dump(self.all_features, outfile, indent=2)

        print(f"Info: Saved all features to '{json_path}'.")

    def eventloop(self):

        print("======================================")
        print("|| Multiple Image Feature Selection ||")
        print("======================================")

        path_to_images = os.path.join(self.path_to_objects, "images")

        files = os.listdir(path_to_images)
        png_files = [file for file in files if file.lower().endswith('.png')]

        for file in png_files:

            print(f"Info: Feature selection for '{file}'.")

            path_to_image = os.path.join(path_to_images, file)
            features = self.feature_picker.eventloop(path_to_image)
            print(f"Info: Selected features: {features}")

            # Save features for this image
            self._create_json(file, features)

            # Store in combined dict
            self.all_features[file] = features

        # Save combined features file
        self._save_all_features()

        return self.all_features

