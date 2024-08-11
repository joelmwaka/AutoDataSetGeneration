import os
import cv2
import random
import logging


class Source:

    def __init__(self, source_type: str):

        self.source_type = source_type

    def __len__(self):

        raise NotImplementedError

    def get_background_image(self):

        raise NotImplementedError


class ImageBGSource(Source):

    def __init__(self, path_images):

        logging.debug("Initializing image background source.")

        source_type = "image"
        super().__init__(source_type)

        self.path_images = path_images
        self.list_image_file_names = []

        if os.path.isdir(self.path_images):
            for file_name in os.listdir(self.path_images):
                if file_name.split(".")[-1].lower() in ["jpeg", "jpg", "png", "tiff"]:
                    self.list_image_file_names.append(file_name)
        else:
            raise NotADirectoryError(f"{self.path_images} is not a directory.")

        if not len(self):
            raise FileNotFoundError("No image files were found.")

        logging.debug("Done initializing image background source.")

    def __len__(self):

        return len(self.list_image_file_names)

    def get_background_image(self):

        file_name = random.choice(self.list_image_file_names)
        file_path = os.path.join(self.path_images, file_name)

        logging.debug(f"Background image: {file_path}")

        return cv2.imread(file_path) if os.path.exists(file_path) else None


class VideoBGSource(Source):

    def __init__(self, path_videos):

        logging.debug("Initializing video background source.")

        source_type = "video"
        super().__init__(source_type)

        self.path_videos = path_videos
        self.list_video_file_names = []

        if os.path.isdir(self.path_videos):
            for file_name in os.listdir(self.path_videos):
                if file_name.split(".")[-1].lower() in ["mp4", "avi", "mkv", "mov", "wmv"]:
                    self.list_video_file_names.append(file_name)
        else:
            raise NotADirectoryError(f"{self.path_videos} is not a directory.")

        if not len(self):
            raise FileNotFoundError("No video files were found.")

        logging.debug("Done initializing video background source.")

    def __len__(self):

        return len(self.list_video_file_names)

    def get_background_image(self):

        video_file_name = random.choice(self.list_video_file_names)
        video_file_path = os.path.join(self.path_videos, video_file_name)

        if os.path.exists(video_file_path):
            cap = cv2.VideoCapture(video_file_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_file_path}")

            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not num_frames:
                raise ValueError("The video file has no frames.")

            random_frame_index = random.randint(0, num_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)

            success, frame = cap.read()

            if not success:
                raise ValueError("Could not read the frame from the video.")

            cap.release()

            logging.debug(f"Background image: Frame {random_frame_index} from {video_file_path}")

            return frame


class BackgroundImage:

    def __init__(self, meta_data, image):

        self.meta_data = meta_data
        self.image = image


class Background:

    def __init__(self, settings: dict):

        self.settings = settings

        self.path_imgs = self.settings["sources"]["images"] \
            if self.check_source_setting_exists("images") else None
        self.path_vids = self.settings["sources"]["videos"] \
            if self.check_source_setting_exists("videos") else None

        if self.path_imgs is None and self.path_vids is None:
            raise FileNotFoundError("No background source files given in settings.yaml. "
                                    "See README for instructions on how to define "
                                    "sources for the background images.")

        self.sources = []

        if self.path_imgs:
            self.sources.append(ImageBGSource(self.path_imgs))

        if self.path_vids:
            self.sources.append(VideoBGSource(self.path_vids))

    def check_source_setting_exists(self, source):

        return True if source in self.settings["sources"].keys() else False

    def get_background_image(self) -> BackgroundImage:

        source = random.choice(self.sources)
        image = source.get_background_image()
        h, w, c = image.shape

        return BackgroundImage({"height": h, "width": w, "channels": c}, image)
