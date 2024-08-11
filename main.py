import logging
from generator import Generator

PATH_YAML = "settings.yaml"

logging.basicConfig(
    level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():

    generator = Generator(path_settings=PATH_YAML)
    generator.run()


if __name__ == "__main__":
    main()
