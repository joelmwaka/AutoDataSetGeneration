import os
import json

from utils.feature_picker import MultipleImageFeaturePicker


PATH_TO_CONFIGS = "./configs/configurations.json"

def main():

    # configuration
    config = {}
    try:
        with open(PATH_TO_CONFIGS, "r") as file:
            config = json.load(file)
    except Exception as e:
        print(f"Error: {e}")

    steps = config["step_selection"]
    data = config["data"]
    
    # step 1: annotate the object images
    if steps["step_1"]:

        path_to_objects = data["path_objects"]
        path_to_object_images = os.path.join(path_to_objects, "images")
        path_to_object_annotations = os.path.join(path_to_objects, "annotations")

        feature_picker = MultipleImageFeaturePicker(path_to_objects=path_to_objects)

        # check if object images folder exists
        if not os.path.exists(path_to_object_images):
            print(f"Error: Folder {path_to_object_images} does not exist.")
            return
        
        # get a list of all files in the folder
        files = os.listdir(path_to_object_images)
        png_files = [file for file in files if file.lower().endswith('.png')]

        # rename all png files with integers
        for i, png_file in enumerate(png_files):
            new_name = f"{i}.png"
            old_path = os.path.join(path_to_object_images, png_file)
            new_path = os.path.join(path_to_object_images, new_name)
            os.rename(old_path, new_path)
            print(f"Info: Renamed '{png_file}' to '{new_name}'.")

        # pick features
        feature_picker.eventloop()
        

    # step 2: get background images from videos
    if steps["step_2"]:

        pass

    # Step 3: ...


if __name__ == "__main__":
    main()