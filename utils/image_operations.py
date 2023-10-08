import cv2
import copy
import random
import numpy as np
import sys

utils_module = "DataGeneration/utils"
sys.path.append(utils_module)


class AlterOriginalImagePose:

    def __init__(self, image: np.ndarray, feature_points: dict, bounding_box: dict, config_params: dict):

        # Miscellaneous Variables
        self.original_image = image
        self.orig_h, self.orig_w = self.original_image.shape[:2]
        self.orig_feature_points = feature_points
        self.orig_bounding_box = bounding_box
        self.config = config_params
        self.thickness = copy.deepcopy(self.config["thickness"])
        self.viewing_distance = copy.deepcopy(self.config["viewing_distance"])
        self.range_angles = copy.deepcopy(self.config["range_angles"])
        self.range_scale_factor = copy.deepcopy(self.config["range_scale"])
        self.point = (self.thickness, self.thickness)

        # Original Image Corners
        self.image_corners = {
            "top_left": [0, 0],
            "top_right": [self.orig_w, 0],
            "bottom_right": [self.orig_w, self.orig_h],
            "bottom_left": [0, self.orig_h]
        }

        # Collected Pipeline Images
        self.bordered_image = None
        self.point_rotated_image = None
        self.line_rotated_image = None
        self.scaled_image = None
        self.cropped_image = None

        # Background mask
        self.mask = None

        # Homography computation points
        self.homo_tf_points = {
            "orig_plane": [],
            "trans_plane": []
        }

        # Feature Points and Bounding Box
        self.altered_feature_points = self.orig_feature_points
        self.altered_bounding_box = self.orig_bounding_box

        # DEBUGGING
        if self.config["show_images_object_pose_alteration"]:
            self.display_annotated_image(self.original_image)

    def draw_image_boundary(self, image: np.ndarray):

        boundary_color = (0, 0, 255)

        image_drawn = cv2.line(image, [int(x) for x in self.image_corners["top_left"]],
                               [int(x) for x in self.image_corners["top_right"]], boundary_color, 2)
        image_drawn = cv2.line(image_drawn, [int(x) for x in self.image_corners["top_right"]],
                               [int(x) for x in self.image_corners["bottom_right"]], boundary_color, 2)
        image_drawn = cv2.line(image_drawn, [int(x) for x in self.image_corners["bottom_right"]],
                               [int(x) for x in self.image_corners["bottom_left"]], boundary_color, 2)
        image_drawn = cv2.line(image_drawn, [int(x) for x in self.image_corners["bottom_left"]],
                               [int(x) for x in self.image_corners["top_left"]], boundary_color, 2)

        return image_drawn

    def draw_features_in_image(self, image: np.ndarray):

        point_color = (0, 255, 0)
        image_drawn = np.copy(image)

        for key in self.altered_feature_points:
            cv2.circle(image_drawn, [int(x) for x in self.altered_feature_points[key]], 4, point_color, -1)

        return image_drawn

    def draw_rotation_line(self, image: np.ndarray):

        line_color = (255, 0, 0)

        image_drawn = cv2.line(image, [int(x) for x in self.image_corners["top_left"]],
                               [int(x) for x in self.image_corners["bottom_left"]], line_color, 2)

        return image_drawn

    def display_annotated_image(self, image: np.ndarray, details=True):

        if details:
            image_draw = self.draw_image_boundary(image=image)
            image_draw = self.draw_features_in_image(image=image_draw)
            image_draw = self.draw_rotation_line(image=image_draw)
        else:
            image_draw = np.copy(image)

        cv2.imshow('Image', image_draw)
        cv2.waitKey(0)  # Wait for any key press
        cv2.destroyAllWindows()

    def expand_image(self):

        # Get the dimensions of the original image
        h, w = self.original_image.shape[:2]

        # Create a blank image
        bordered_image = np.zeros((h + 2 * self.thickness, w + 2 * self.thickness, 3), dtype=np.uint8)

        # Paste the original image
        bordered_image[self.thickness:self.thickness + h, self.thickness:self.thickness + w] = self.original_image

        # Alter image corners
        for key in self.image_corners:
            old_point = self.image_corners[key]
            self.image_corners[key] = [old_point[0] + self.thickness, old_point[1] + self.thickness]

        # Alter all features and bounding boxes
        for key in self.altered_feature_points:
            old_point = self.altered_feature_points[key]
            self.altered_feature_points[key] = [old_point[0] + self.thickness, old_point[1] + self.thickness]
        for key in self.altered_bounding_box:
            old_point = self.altered_bounding_box[key]
            self.altered_bounding_box[key] = [old_point[0] + self.thickness, old_point[1] + self.thickness]

        # DEBUGGING
        if self.config["show_images_object_pose_alteration"]:
            self.display_annotated_image(image=bordered_image)

        self.bordered_image = bordered_image

    def rotate_image_around_top_left_corner(self, angle_deg: float):

        h, w = self.line_rotated_image.shape[:2]

        # Calculate rotation matrix and rotate image
        rotation_matrix = cv2.getRotationMatrix2D(center=self.point, angle=angle_deg, scale=1.0)
        rotated_image = cv2.warpAffine(src=self.line_rotated_image, M=rotation_matrix, dsize=(w, h),
                                       flags=cv2.INTER_LINEAR)

        # Alter image corners
        for key in self.image_corners:
            old_point = self.image_corners[key]
            old_point.append(1)
            new_point = np.dot(rotation_matrix, np.array(old_point))
            self.image_corners[key] = [new_point[0], new_point[1]]

        # Alter all features and bounding boxes
        for key in self.altered_feature_points:
            old_point = self.altered_feature_points[key]
            old_point.append(1)
            new_point = np.dot(rotation_matrix, np.array(old_point))
            self.altered_feature_points[key] = [new_point[0], new_point[1]]
        for key in self.altered_bounding_box:
            old_point = self.altered_bounding_box[key]
            old_point.append(1)
            new_point = np.dot(rotation_matrix, np.array(old_point))
            self.altered_bounding_box[key] = [new_point[0], new_point[1]]

        # DEBUGGING
        if self.config["show_images_object_pose_alteration"]:
            self.display_annotated_image(image=rotated_image)

        self.point_rotated_image = rotated_image

    def rotate_image_around_left_edge(self, angle_deg: float):

        h, w = self.bordered_image.shape[:2]

        # Get homography points from original image - take corners
        for key in self.image_corners:
            self.homo_tf_points["orig_plane"].append(self.image_corners[key])

        # Get homography points in transformed image - take corners
        apparent_width_of_image = self.orig_w * np.cos(np.deg2rad(angle_deg))
        delta_x = self.orig_w * np.tan(np.deg2rad(angle_deg))
        apparent_height_rotated_edge = self.orig_h * (np.arctan(self.orig_h / (self.viewing_distance - delta_x)) /
                                                      np.arctan(self.orig_h / self.viewing_distance))
        delta_height = (apparent_height_rotated_edge - self.orig_h) / 2

        for key in self.image_corners:
            if key in ["top_left", "bottom_left"]:
                self.homo_tf_points["trans_plane"].append(self.image_corners[key])
            elif key == "top_right":
                self.homo_tf_points["trans_plane"].append([self.image_corners["top_left"][0] + apparent_width_of_image,
                                                           self.image_corners[key][1] - delta_height])
            elif key == "bottom_right":
                self.homo_tf_points["trans_plane"].append(
                    [self.image_corners["bottom_left"][0] + apparent_width_of_image,
                     self.image_corners[key][1] + delta_height])

        # Compute homography matrix
        src_points = np.array(self.homo_tf_points["orig_plane"], dtype=np.float32)
        dst_points = np.array(self.homo_tf_points["trans_plane"], dtype=np.float32)
        homography_matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        rotated_image = cv2.warpPerspective(self.bordered_image, homography_matrix, (w, h))

        # Alter image corners
        for key in self.image_corners:
            old_point = np.array(self.image_corners[key], dtype=np.float32).reshape(-1, 1, 2)
            new_point = cv2.perspectiveTransform(old_point, homography_matrix)
            self.image_corners[key] = [new_point[0][0][0], new_point[0][0][1]]

        # Alter all features and bounding boxes
        for key in self.altered_feature_points:
            old_point = np.array(self.altered_feature_points[key], dtype=np.float32).reshape(-1, 1, 2)
            new_point = cv2.perspectiveTransform(old_point, homography_matrix)
            self.altered_feature_points[key] = [new_point[0][0][0], new_point[0][0][1]]
        for key in self.altered_bounding_box:
            old_point = np.array(self.altered_bounding_box[key], dtype=np.float32).reshape(-1, 1, 2)
            new_point = cv2.perspectiveTransform(old_point, homography_matrix)
            self.altered_bounding_box[key] = [new_point[0][0][0], new_point[0][0][1]]

        # DEBUGGING
        if self.config["show_images_object_pose_alteration"]:
            self.display_annotated_image(rotated_image)

        self.line_rotated_image = rotated_image

    def scale_image(self, scale_factor: float):

        if scale_factor <= 0:
            raise ValueError("Scale factor must be greater than 0")

        h, w = self.point_rotated_image.shape[:2]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        scaled_image = cv2.resize(self.point_rotated_image, (new_w, new_h))

        # Alter image corners
        for key in self.image_corners:
            old_point = self.image_corners[key]
            self.image_corners[key] = [old_point[0] * scale_factor, old_point[1] * scale_factor]

        # Alter all features and bounding boxes
        for key in self.altered_feature_points:
            old_point = self.altered_feature_points[key]
            self.altered_feature_points[key] = [old_point[0] * scale_factor, old_point[1] * scale_factor]
        for key in self.altered_bounding_box:
            old_point = self.altered_bounding_box[key]
            self.altered_bounding_box[key] = [old_point[0] * scale_factor, old_point[1] * scale_factor]

        # DEBUGGING
        if self.config["show_images_object_pose_alteration"]:
            self.display_annotated_image(scaled_image)

        self.scaled_image = scaled_image

    def crop_tight_box(self):

        # Get dimensions of tight object image
        row_points = []
        col_points = []
        for key in self.image_corners:
            col_points.append(int(self.image_corners[key][0]))
            row_points.append(int(self.image_corners[key][1]))

        min_row, max_row = (min(row_points), max(row_points))
        min_col, max_col = (min(col_points), max(col_points))

        cropped_image = self.scaled_image[min_row:max_row, min_col:max_col]

        # Alter image corners
        for key in self.image_corners:
            old_point = self.image_corners[key]
            self.image_corners[key] = [old_point[0] - min_col, old_point[1] - min_row]

        # Alter all features and bounding boxes
        for key in self.altered_feature_points:
            old_point = self.altered_feature_points[key]
            self.altered_feature_points[key] = [old_point[0] - min_col, old_point[1] - min_row]
        for key in self.altered_bounding_box:
            old_point = self.altered_bounding_box[key]
            self.altered_bounding_box[key] = [old_point[0] - min_col, old_point[1] - min_row]

        # DEBUGGING
        if self.config["show_images_object_pose_alteration"]:
            self.display_annotated_image(image=cropped_image)

        self.cropped_image = cropped_image

    def create_mask(self):

        vertices = []
        h, w = self.cropped_image.shape[:2]
        self.mask = np.zeros(shape=(h, w), dtype=np.int8)

        for key in self.image_corners:
            vertices.append([int(self.image_corners[key][0]), int(self.image_corners[key][1])])
        vertices = np.array(vertices, dtype=np.int32)

        cv2.fillPoly(self.mask, [vertices], 255)

        # DEBUGGING
        if self.config["show_images_object_pose_alteration"]:
            self.display_annotated_image(image=self.mask, details=False)

    def choose_random_angle(self):

        mean = (self.range_angles[0] + self.range_angles[1]) / 2
        std_deviation = (self.range_angles[1] - mean) / 2

        random_value = np.random.normal(mean, std_deviation)
        random_value = max(min(random_value, self.range_angles[1]), self.range_angles[0])

        return random_value

    def choose_random_scale_factor(self):

        random_value = np.random.uniform(low=self.range_scale_factor[0], high=self.range_scale_factor[1])

        return random_value

    def eventloop(self):

        angle_edge_rotation = self.choose_random_angle()
        angle_corner_rotation = self.choose_random_angle()
        scale_factor = self.choose_random_scale_factor()

        self.expand_image()
        self.rotate_image_around_left_edge(angle_deg=angle_edge_rotation)
        self.rotate_image_around_top_left_corner(angle_deg=angle_corner_rotation)
        self.scale_image(scale_factor=scale_factor)
        self.crop_tight_box()
        self.create_mask()

        return {
            "output_image": self.cropped_image,
            "corners": self.image_corners,
            "features": self.altered_feature_points,
            "boundbox": self.altered_bounding_box,
            "mask": self.mask
        }


class CreateSingleAnnotatedImage:

    def __init__(self, config_params: dict, image_size="large"):

        self.config = config_params

        assert image_size in ["small", "large"]

        if image_size == "large":
            shape = copy.deepcopy(self.config["large_final_image_shape"])
            self.final_image_w, self.final_image_h = (shape[0], shape[1])
        elif image_size == "small":
            shape = copy.deepcopy(self.config["small_final_image_shape"])
            self.final_image_w, self.final_image_h = (shape[0], shape[1])

        self.num_objects = 0

        self.image_manipulator = ImageAlteration(config=copy.deepcopy(self.config))

        # To be configured before eventloop
        self.object_image_operator = None
        self.background_image = None
        self.bg_image_h, self.bg_image_w = (None, None)

        # Miscellaneous Variables
        self.altered_data = None
        self.object_image = None
        self.oj_image_h = None
        self.oj_image_w = None
        self.object_corners = None
        self.object_features = None
        self.object_boundbox = {}
        self.list_object_features = []
        self.list_object_boundboxes = []
        self.object_mask = None

        # Images collected in pipeline
        self.image_with_object = None

    def initialize(self, object_image_data: dict, background_image: np.ndarray):

        self.object_corners = None
        self.object_features = None
        self.object_boundbox = {}

        self.image_manipulator.initialize_random_params()

        self.object_image_operator = AlterOriginalImagePose(
            image=copy.deepcopy(object_image_data["object_image"]),
            feature_points=copy.deepcopy(object_image_data["features"]),
            bounding_box=copy.deepcopy(object_image_data["boundbox"]),
            config_params=copy.deepcopy(self.config)
        )

        self.altered_data = self.object_image_operator.eventloop()
        self.get_data_from_altered_object_image()

        self.background_image = background_image
        self.bg_image_h, self.bg_image_w = self.background_image.shape[:2]

    def draw_image_boundary(self, image: np.ndarray):

        boundary_color = (0, 0, 255)

        image_drawn = cv2.line(image, [int(x) for x in self.object_corners["top_left"]],
                               [int(x) for x in self.object_corners["top_right"]], boundary_color, 2)
        image_drawn = cv2.line(image_drawn, [int(x) for x in self.object_corners["top_right"]],
                               [int(x) for x in self.object_corners["bottom_right"]], boundary_color, 2)
        image_drawn = cv2.line(image_drawn, [int(x) for x in self.object_corners["bottom_right"]],
                               [int(x) for x in self.object_corners["bottom_left"]], boundary_color, 2)
        image_drawn = cv2.line(image_drawn, [int(x) for x in self.object_corners["bottom_left"]],
                               [int(x) for x in self.object_corners["top_left"]], boundary_color, 2)

        return image_drawn

    @staticmethod
    def draw_boundbox_in_image(image: np.ndarray, bbox: dict):

        color = (0, 0, 255)

        top_left = bbox["top_left"]
        bottom_right = bbox["bottom_right"]

        image_drawn = cv2.rectangle(image, top_left, bottom_right, color, 2)

        return image_drawn

    @staticmethod
    def draw_features_in_image(image: np.ndarray, features: dict):

        point_color = (0, 255, 0)
        image_drawn = np.copy(image)

        for key in features:
            cv2.circle(image_drawn, [int(x) for x in features[key]], 4, point_color, -1)

        return image_drawn

    def display_annotated_image(self, image: np.ndarray, features=False, boundbox=False):

        if features or boundbox:
            image_draw = np.copy(image)
            if features:
                for feats in self.list_object_features:
                    image_draw = self.draw_features_in_image(image=image_draw, features=feats)
            if boundbox:
                for bbox in self.list_object_boundboxes:
                    image_draw = self.draw_boundbox_in_image(image=image_draw, bbox=bbox)
        else:
            image_draw = np.copy(image)

        cv2.imshow('Image', image_draw)
        cv2.waitKey(0)  # Wait for any key press
        cv2.destroyAllWindows()

    def get_data_from_altered_object_image(self):

        self.object_image = copy.deepcopy(self.altered_data["output_image"])
        self.oj_image_h, self.oj_image_w = self.object_image.shape[:2]
        self.object_corners = copy.deepcopy(self.altered_data["corners"])
        self.object_features = copy.deepcopy(self.altered_data["features"])
        self.object_mask = copy.deepcopy(self.altered_data["mask"])

    def random_resize(self):

        scale = np.random.uniform(low=0.8, high=1.2)
        new_width = int(self.bg_image_w * scale)
        new_height = int(self.bg_image_h * scale)

        self.background_image = cv2.resize(self.background_image, (new_width, new_height))
        self.bg_image_h, self.bg_image_w = self.background_image.shape[:2]

    def get_final_background_image(self):

        while True:
            if self.bg_image_h > self.final_image_h and self.bg_image_w > self.final_image_w:

                # Choose random point to start crop from in background image
                point = [
                    int(np.random.uniform(low=0, high=self.bg_image_w - self.final_image_w)),
                    int(np.random.uniform(low=0, high=self.bg_image_h - self.final_image_h))
                ]

                # Crop final background image
                final_background_image = self.background_image[
                                         point[1]:point[1] + self.final_image_h,
                                         point[0]:point[0] + self.final_image_w
                                         ]

                # Update variables for next steps in this class
                self.background_image = np.copy(final_background_image)
                self.bg_image_h, self.bg_image_w = self.background_image.shape[:2]

                break

            else:

                # Enlarge image by scale 1.5
                new_width = int(self.bg_image_w * 1.5)
                new_height = int(self.bg_image_h * 1.5)

                self.background_image = cv2.resize(self.background_image, (new_width, new_height))
                self.bg_image_h, self.bg_image_w = self.background_image.shape[:2]

                # print(self.bg_image_h)

    def adjust_object_image_darkness(self):

        mean_brightness_background = np.mean(self.background_image)
        alpha_object_image = self.object_mask / 127
        object_image = np.copy(self.object_image)
        for c in range(0, 3):
            object_image[:, :, c] = alpha_object_image * self.object_image[:, :, c]
        mean_brightness_object = np.mean(object_image)

        adjustment_factor = mean_brightness_background / mean_brightness_object
        self.object_image = cv2.convertScaleAbs(self.object_image, alpha=adjustment_factor, beta=0)

    def add_object_to_bg_image(self):

        # Copy background image
        image = np.copy(self.background_image)

        alpha_object_image = self.object_mask / 127
        alpha_background_image = 1 - alpha_object_image

        # Choose random point to input object in background image
        point = [
            int(np.random.uniform(low=0, high=self.bg_image_w - self.oj_image_w)),
            int(np.random.uniform(low=0, high=self.bg_image_h - self.oj_image_h))
        ]

        # Insert object into background image
        for c in range(0, 3):
            background_image_cutout = image[point[1]:point[1] + self.oj_image_h, point[0]:point[0] + self.oj_image_w, c]
            image[point[1]:point[1] + self.oj_image_h, point[0]:point[0] + self.oj_image_w, c] \
                = (alpha_object_image * self.object_image[:, :, c] +
                   alpha_background_image * background_image_cutout)

        # Correct corner and features  points
        for key in self.object_corners:
            old_point = self.object_corners[key]
            self.object_corners[key] = [old_point[0] + point[0], old_point[1] + point[1]]
        for key in self.object_features:
            old_point = self.object_features[key]
            self.object_features[key] = [old_point[0] + point[0], old_point[1] + point[1]]

        self.image_with_object = image

    def get_bound_box(self):

        # Get x and y points from corners of object image
        h_list = []
        w_list = []

        for key in self.object_corners:
            w_list.append(self.object_corners[key][0])
            h_list.append(self.object_corners[key][1])

        self.object_boundbox["top_left"] = [int(min(w_list)), int(min(h_list))]
        self.object_boundbox["bottom_right"] = [int(max(w_list)), int(max(h_list))]

    def eventloop(self, object_image_data: dict, background_image: np.ndarray, num_objects: int):

        updated_background_image = np.copy(background_image)
        self.num_objects = num_objects
        self.list_object_features = []
        self.list_object_boundboxes = []

        if num_objects == 0:

            # Initialize object and background image
            self.initialize(object_image_data=object_image_data, background_image=updated_background_image)

            # Random resize of original background image
            self.random_resize()

            # Get final background image
            self.get_final_background_image()

            # Save final image
            self.image_with_object = self.background_image

        else:
            for i in range(self.num_objects):

                # Initialize object and background image
                self.initialize(object_image_data=object_image_data, background_image=updated_background_image)

                # Get final background image
                if i == 0:
                    # Random resize of original background image
                    self.random_resize()
                    # Get final background image
                    self.get_final_background_image()
                else:
                    self.background_image = np.copy(self.image_with_object)

                # Adjust object image brightness
                self.adjust_object_image_darkness()

                # Add object to image
                self.add_object_to_bg_image()

                # Get new bound box
                self.get_bound_box()

                # Collect features and bounding boxes
                features = copy.deepcopy(self.object_features)
                self.list_object_features.append(features)
                self.list_object_boundboxes.append(self.object_boundbox)

                updated_background_image = np.copy(self.image_with_object)

        # Do image altering operations on image
        self.image_with_object = self.image_manipulator.eventloop(image_bgr=self.image_with_object)

        # self.display_annotated_image(self.image_with_object, features=True, boundbox=True)

        return {
            "image": self.image_with_object,
            "boundboxes": self.list_object_boundboxes,
            "features": self.list_object_features
        }


class ImageAlteration:

    def __init__(self, config: dict):

        self.config = config["image_enhancement"]

        # NOISE Parameters
        self.k = None

        # GAUSSIAN Parameters
        self.gaussian_kernel_size = None
        self.gaussian_sigma = None

        # LAPLACIAN Parameters
        self.laplacian_kernel_size = None

        # BRIGHTNESS Parameters
        self.brightness_alpha = None
        self.brightness_beta = None

        # SATURATION Parameters
        self.saturation_factor = None

        # Operation params
        self.standard_operations = False
        self.num_standard_operations = 0
        self.standard_operations_list = []
        self.further_operations = False
        self.num_further_operations = 0
        self.further_operations_list = []

    def initialize_random_params(self):

        # NOISE Parameters
        k_min = self.config["noise"]["k"][0]
        k_max = self.config["noise"]["k"][1]
        self.k = np.random.uniform(low=k_min, high=k_max)

        # GAUSSIAN Parameters
        k_size = random.choice(self.config["gaussian_smoothing"]["kernel_size"])
        self.gaussian_kernel_size = [k_size, k_size]
        self.gaussian_sigma = self.config["gaussian_smoothing"]["sigma"]

        # LAPLACIAN Parameters
        self.laplacian_kernel_size = self.config["laplacian_edge_enhancement"]["kernel_size"]

        # BRIGHTNESS Parameters
        alpha_min = self.config["brightness"]["alpha"][0]
        alpha_max = self.config["brightness"]["alpha"][1]
        self.brightness_alpha = 2.0  # np.random.uniform(low=alpha_min, high=alpha_max)
        self.brightness_beta = self.config["brightness"]["beta"]

        # SATURATION Parameters
        factor_min = self.config["saturation"]["factor"][0]
        factor_max = self.config["saturation"]["factor"][1]
        self.saturation_factor = np.random.uniform(low=factor_min, high=factor_max)

        # Standard operations settings
        self.standard_operations = random.choice([True, False])
        all_standard_operations = ["smoothing", "brightness", "saturation"]  # "noise", "sharpening"
        self.num_standard_operations = random.randint(1, len(all_standard_operations))
        self.standard_operations_list = random.sample(all_standard_operations, self.num_standard_operations)

        # Further operations settings
        self.further_operations = False  # not self.standard_operations
        all_further_operations = ["shadows", "rainy", "foggy", "flares", "speedy"]
        self.num_further_operations = random.randint(1, len(all_further_operations))
        self.further_operations_list = random.sample(all_further_operations, self.num_further_operations)

    def add_noise(self, image_bgr):  # Operation 1

        if self.k < 0:
            raise ValueError("Scaling factor k must be greater than or equal to 0.")

        std_dev = self.k * 3  # You can adjust this value to control noise intensity
        noise = np.random.normal(0, std_dev, image_bgr.shape).astype(np.uint8)
        noisy_image = cv2.add(image_bgr, noise)
        noisy_image = np.clip(noisy_image, 0, 255)

        return noisy_image

    def gaussian_smoothing(self, image_bgr):  # Operation 2

        if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Input image must be a valid BGR image (3 channels).")

        # Apply Gaussian blur
        image_smoothened = cv2.GaussianBlur(image_bgr, self.gaussian_kernel_size, self.gaussian_sigma)

        return image_smoothened

    def laplacian_sharpening(self, image_bgr):  # Operation 3

        if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Input image must be a valid BGR image (3 channels).")

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Apply the Laplacian filter for edge detection
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=self.laplacian_kernel_size)
        # Convert the Laplacian result back to uint8
        laplacian = cv2.convertScaleAbs(laplacian)
        # Stack the Laplacian image with the original image to maintain the BGR format
        laplacian_image_bgr = cv2.merge((laplacian, laplacian, laplacian))
        # Sharpen the image by adding the Laplacian result to the original image
        image_sharpened = cv2.addWeighted(image_bgr, 1.0, laplacian_image_bgr, 0.01, 0)

        return image_sharpened

    def add_brightness(self, image_bgr):  # Operation 4

        adjusted_image = cv2.convertScaleAbs(image_bgr, alpha=self.brightness_alpha, beta=self.brightness_beta)

        return adjusted_image

    def adjust_saturation(self, image_bgr):  # Operation 5

        if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Input image must be a valid BGR image (3 channels).")

        # Transform BGR image to HSV
        hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        # Alter saturation
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * self.saturation_factor, 0, 255)
        image_saturated = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return image_saturated

    def eventloop(self, image_bgr):

        self.initialize_random_params()

        if np.random.uniform() < 0.7:

            if self.standard_operations:
                for operation in self.standard_operations_list:
                    if operation == "noise":
                        image_bgr = self.add_noise(image_bgr)
                    elif operation == "smoothing":
                        image_bgr = self.gaussian_smoothing(image_bgr)
                    elif operation == "sharpening":
                        image_bgr = self.laplacian_sharpening(image_bgr)
                    elif operation == "brightness":
                        image_bgr = self.add_brightness(image_bgr)
                    elif operation == "saturation":
                        image_bgr = self.adjust_saturation(image_bgr)

            if self.further_operations:
                for operation in self.further_operations_list:
                    if operation == "shadows":
                        image_bgr = add_shadow(image_bgr, no_of_shadows=1)
                    elif operation == "snowy":
                        image_bgr = add_snow(image_bgr, snow_coeff=0.3)
                    elif operation == "rainy":
                        image_bgr = add_rain(image_bgr, slant=random.randint(0, 20), rain_type="heavy")
                    elif operation == "foggy":
                        image_bgr = add_fog(image_bgr, fog_coeff=0.3)
                    elif operation == "speedy":
                        image_bgr = add_speed(image_bgr, speed_coeff=0.2)

        return image_bgr

