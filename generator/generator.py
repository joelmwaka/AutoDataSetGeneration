import traceback

from generator.settings import Settings
from generator.objects import Objects
from generator.background import Background
from generator.alteration import Alteration


class Generator:

    def __init__(self, path_settings):

        self.path_settings = path_settings
        self.settings = Settings(path_settings=path_settings)
        self.number_of_samples = self.settings.generator["number_of_samples"]
        self.maximum_objects_per_image = self.settings.generator["maximum_objects_per_image"]
        self.objects = None

    def run(self):

        try:

            # initialize objects and background
            objects = Objects(self.settings.objects)
            background = Background(self.settings.background)

            counter = 0

            while counter < self.number_of_samples:

                bgd = background.get_background_image()
                print(bgd.image)
                print(bgd.meta_data)

                obj = objects.get_object()
                print(obj.label)
                print(obj.bounding_box)
                print(obj.num_features)
                print(obj.features)

                # TODO:

                counter += 1

            print(counter)

            # generate dataset

        except Exception as e:
            traceback.print_exc()
            print(f"Exception: {e}")
