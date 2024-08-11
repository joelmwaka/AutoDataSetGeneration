import yaml


class Settings:

    def __init__(self, path_settings):

        self.path_settings = path_settings

        self.objects = None
        self.background = None
        self.alteration = None
        self.generator = None

        self.configure()

    def configure(self):

        with open(self.path_settings, "r") as file:
            settings = yaml.safe_load(file)

        self.objects = settings["objects"]
        self.background = settings["background"]
        self.alteration = settings["alteration"]
        self.generator = settings["generator"]

