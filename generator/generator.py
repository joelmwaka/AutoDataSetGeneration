from generator.utils.settings import Settings
from generator.utils.objects import Objects
from generator.utils.background import Background
from generator.utils.alteration import Alteration


class Generator:

    def __init__(self, path_settings):

        self.path_settings = path_settings
        self.settings = Settings(path_settings=path_settings)
        self.objects = None

    def eventloop(self):

        # try:

        # annotate and save objects data
        objs = Objects(self.settings.objects)
        self.objects = objs.eventloop()

        # initialize background
        background = Background(self.settings.background)

        # alteration
        alteration = Alteration(self.settings.alteration)
        alteration.eventloop(self.objects[0])

        # generate dataset

        # except ValueError as v:
        #     print(f"Value Error: {v}")
        #
        # except Exception as e:
        #     print(f"Exception: {e}")


