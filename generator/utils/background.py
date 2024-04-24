import os
import cv2


class Background:

    def __init__(self, settings: dict):

        self.settings = settings

        self.path_images = self.settings["sources"]["images"]
        self.path_videos = self.settings["sources"]["videos"]
        self.file_names = [fname for fname in os.listdir(self.path_images) if fname.lower().endswith(".png")]
        self.num_images = len(self.file_names)

        self.pointer = 0

    def get_background_image(self):

        if self.pointer <= self.num_images - 1:
            path_image = os.path.join(self.path_images, self.file_names[self.pointer])
            image = cv2.imread(path_image)
            height, width, _ = image.shape

            return (height, width), image

        else:
            print(f"Info: No more background images!")

            return (0, 0), None
