import os
import cv2


def get_frames_from_video(path_to_video, path_to_folder, prefix, step=5):

    print("INFO: Getting images from '" + path_to_video + "'.")
    cap = cv2.VideoCapture(path_to_video)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % step == 0:
            file_name = prefix + str(int(frame_count/step)) + ".png"
            path_to_image = path_to_folder + file_name
            cv2.imwrite(path_to_image, frame)

    print("INFO: Done saving images.")
    cap.release()


def rename_all_images(path_to_folder):

    files = os.listdir(path_to_folder)

    for index, file in enumerate(files):
        old_file_path = os.path.join(path_to_folder, file)
        new_file_path = os.path.join(path_to_folder, str(index)+".png")
        os.rename(old_file_path, new_file_path)
