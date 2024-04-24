from generator.generator import Generator

PATH_YAML = "settings.yaml"


def main():

    generator = Generator(path_settings=PATH_YAML)

    generator.eventloop()


if __name__ == "__main__":
    main()
