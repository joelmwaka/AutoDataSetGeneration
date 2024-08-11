import os
import cv2


class Background:

    def __init__(self, settings: dict):

        self.settings = settings

        # get source settings
        self.path_images = self.settings["sources"]["images"] if self.check_source_setting_exists("images") else None
        self.path_videos = self.settings["sources"]["videos"] if self.check_source_setting_exists("videos") else None
        self.youtube_link = self.settings["sources"]["youtube"] if self.check_source_setting_exists("youtube") else None

        if self.path_images is None and self.path_videos is None and self.youtube_link is None:
            raise FileNotFoundError("No background source files given.")

        # get data of sources
        self.sources_meta_data = []

        if self.path_images:


            self.sources_meta_data.append({
                "source": "images",
                "num_frames": 0,
            })
        if self.path_videos:

            videos = []

            for f_name in os.listdir(self.path_videos):
                video = cv2.VideoCapture(os.path.join(self.path_videos, f_name))
                if not video.isOpened():
                    raise ValueError(f"Error opening video file: {f_name} in {self.path_videos}.")

                frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                videos.append({
                    "path_video": os.path.join(self.path_videos, f_name)
                    "frame_count": frame_count
                })

            self.sources_meta_data.append({
                "source": "videos",
                "videos": videos
            })
        if self.youtube_link
            pass

        self.file_names = [f_name for f_name in os.listdir(self.path_images) if f_name.lower().endswith(".png")]
        self.num_images = len(self.file_names)

        self.pointer = 0

    def check_source_setting_exists(self, source):
        """ check if source setting exists """
        return True if source in self.settings["sources"].keys() else False

    def choose_background_frame(self):
        """ choose which frame to """

    def get_background_image(self):

        if self.pointer <= self.num_images - 1:
            path_image = os.path.join(self.path_images, self.file_names[self.pointer])
            image = cv2.imread(path_image)
            height, width, _ = image.shape

            return (height, width), image

        else:
            print(f"Info: No more background images!")

            return (0, 0), None
