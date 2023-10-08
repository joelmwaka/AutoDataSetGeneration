import cv2
import copy
import json


class BoundBoxPicker():

    def __init__(self, img_input, path_json=None):

        self.input_image = img_input
        self.input_image_copy = copy.deepcopy(self.input_image)
        self.points = []
        self.path_json = path_json
        self.print_done = False

    def _click_event(self, event, x, y, flags, params):

        # Left mouse click - Select Point
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.input_image_copy, (x, y), 3, (0, 0, 255), -1)
            self.points.append([x, y])
            self.print_done = False

        # Right mouse click - Restart Selection
        if event == cv2.EVENT_RBUTTONDOWN:
            self.input_image_copy = copy.deepcopy(self.input_image)
            self.points.clear()
            self.print_done = False

    def _event_loop(self):

        cv2.namedWindow('Bound Box Picker')
        cv2.setMouseCallback('Bound Box Picker', self._click_event)

        while True:

            if not self.print_done:
                if len(self.points) == 0:
                    print("INSTRUCTION: Pick Top Left Point of Bounding Box")
                elif len(self.points) == 1:
                    print("INSTRUCTION: Pick Bottom Right Point of Bounding Box")
                else:
                    print("INSTRUCTION: Press 'ESC' to end!!!")
                self.print_done = True

            cv2.imshow("Bound Box Picker", self.input_image_copy)
            k = cv2.waitKey(1) & 0xFF

            if k == 27:  # If 'ESC' is pressed.
                if len(self.points) == 2:
                    break
                elif len(self.points) < 2:
                    print("NOTE: Atleast 2 points needed.")
                else:
                    print("NOTE: Too many points selected. Right click on image to restart selection.")

        cv2.destroyAllWindows()

    def _create_json(self):

        dict_boundbox = {}

        dict_boundbox['topleftpoint'] = self.points[0]
        dict_boundbox['bottomrightpoint'] = self.points[1]

        with open(self.path_json + "boundbox.json", "w") as outfile:
            json.dump(dict_boundbox, outfile)

    def run(self):

        print("================")
        print("Bound Box Picker")
        print("================")

        self._event_loop()
        if self.path_json is not None:
            self._create_json()

