import cv2
import numpy as np
from copy import deepcopy
from generator.objects import Object


class Alteration:

    def __init__(self, settings: dict):

        self.settings = settings

        self.angles_range = [self.settings["range_angles"]["min"], self.settings["range_angles"]["max"]]
        self.scale_range = [self.settings["range_scale"]["min"], self.settings["range_scale"]["max"]]
        self.viewing_distance = self.settings["viewing_distance"]

        self.image = None
        self.label = None
        self.bbox = None
        self.num_features = None
        self.features = None

        self.orig_h = None
        self.orig_w = None

    def initialize(self, obj: Object):

        self.image = np.copy(obj.image)
        self.label = deepcopy(obj.label)
        self.bbox = np.copy(obj.bounding_box)
        self.num_features = deepcopy(obj.num_features)
        self.features = np.copy(obj.features)

        self.orig_h, self.orig_w, _ = obj.image.shape

    def add_black_frame(self):

        h, w, _ = self.image.shape
        border_width = max(h, w)

        # black background image
        bg_image = np.zeros((h + 2 * border_width, w + 2 * border_width, 3), dtype=np.uint8)

        # paste object image into background
        bg_image[border_width: h + border_width, border_width: w + border_width] = self.image

        # new bound box
        self.bbox += border_width

        # new features
        if self.num_features > 0:
            self.features += border_width

        self.image = np.copy(bg_image)

    def rotate_image_around_left_edge(self):

        angle_deg = np.random.uniform(self.angles_range[0], self.angles_range[1])
        h, w, _ = self.image.shape

        # get homography points from original image - take corners
        apparent_width_of_image = self.orig_w * np.cos(np.deg2rad(angle_deg))
        delta_x = self.orig_w * np.tan(np.deg2rad(angle_deg))
        apparent_height_rotated_edge = self.orig_h * (np.arctan(self.orig_h / (self.viewing_distance - delta_x)) /
                                                      np.arctan(self.orig_h / self.viewing_distance))
        delta_height = (apparent_height_rotated_edge - self.orig_h) / 2

        homography_pts_orig = np.array([
            [self.bbox[1], self.bbox[0]],  # top left
            [self.bbox[1], self.bbox[2]],  # bottom left
            [self.bbox[3], self.bbox[0]],  # top right
            [self.bbox[3], self.bbox[2]]   # bottom right
        ], dtype=float)

        homography_pts_new = np.array([
            [self.bbox[1], self.bbox[0]],  # top left
            [self.bbox[1], self.bbox[2]],  # bottom left
            [self.bbox[1] + apparent_width_of_image, self.bbox[0] - delta_height],  # top right
            [self.bbox[1] + apparent_width_of_image, self.bbox[3] + delta_height]   # bottom right
        ], dtype=float)

        # compute homography matrix
        homography_matrix, _ = cv2.findHomography(homography_pts_orig, homography_pts_new, cv2.RANSAC, 5.0)

        self.image = cv2.warpPerspective(self.image, homography_matrix, (w, h))

        # new bound box
        self.bbox = np.array([
            np.max(np.array([homography_pts_new[0][1], homography_pts_new[2][1]])),
            self.bbox[1],
            np.min(np.array([homography_pts_new[1][1], homography_pts_new[3][1]])),
            self.bbox[1] + apparent_width_of_image
        ])

        # new features
        if self.num_features > 0:
            self.features = cv2.perspectiveTransform(self.features.reshape(-1, 1, 2), homography_matrix)
            self.features = self.features.reshape(-1, 2)

    def rotate_image_around_top_left_corner(self):

        angle_deg = np.random.uniform(self.angles_range[0], self.angles_range[1])
        h, w, _ = self.image.shape

        # calculate rotation matrix and rotate image
        point = [self.bbox[1], self.bbox[0]]
        rotation_matrix = cv2.getRotationMatrix2D(center=point, angle=angle_deg, scale=1.0)
        self.image = cv2.warpAffine(src=self.image, M=rotation_matrix, dsize=(w, h), flags=cv2.INTER_LINEAR)

        # alter bounding box
        tl_point = [self.bbox[1], self.bbox[0], 1.0]
        br_point = [self.bbox[3], self.bbox[2], 1.0]
        tr_point = [self.bbox[3], self.bbox[0], 1.0]
        bl_point = [self.bbox[1], self.bbox[2], 1.0]
        new_tl_point = np.dot(rotation_matrix, np.array(tl_point))
        new_br_point = np.dot(rotation_matrix, np.array(br_point))
        new_tr_point = np.dot(rotation_matrix, np.array(tr_point))
        new_bl_point = np.dot(rotation_matrix, np.array(bl_point))
        all_points = [new_tl_point, new_br_point, new_tr_point, new_bl_point]
        self.bbox[0] = np.min([pt[1] for pt in all_points])
        self.bbox[1] = np.min([pt[0] for pt in all_points])
        self.bbox[2] = np.max([pt[1] for pt in all_points])
        self.bbox[3] = np.max([pt[0] for pt in all_points])

        # def is_black(row=None, col=None):
        #
        #     if col is None:
        #         row_pixels = self.image[row, :]
        #         return not any(row_pixels)
        #
        #     elif row is None:
        #         col_pixels = self.image[:, col]
        #         return not any(col_pixels)
        #
        #     elif row is not None and col is not None:
        #         pixel = self.image[row, col]
        #         return not any(pixel)
        #
        #     else:
        #         return ValueError(f"Expected either row or col inputs.")

        # alter all features and bounding boxes
        for i, feat in enumerate(self.features):
            self.features[i] = np.dot(rotation_matrix, np.array([feat[0], feat[1], 1.0]))

    def eventloop(self, obj: Object):

        self.initialize(obj)

        self.add_black_frame()
        self.rotate_image_around_left_edge()
        self.rotate_image_around_top_left_corner()

        image = self.visualize()

        while True:
            cv2.imshow("Display", image)
            key = cv2.waitKey(0)

            if key == ord("1"):
                break

    def visualize(self):

        image = np.copy(self.image)

        # draw bounding box on image
        cv2.rectangle(image, (int(self.bbox[1]), int(self.bbox[0])), (int(self.bbox[3]), int(self.bbox[2])),
                      color=(0, 255, 0), thickness=3)

        # draw features on image
        for point in self.features:
            cv2.circle(image, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 255), thickness=-1)

        return image

