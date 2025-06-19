import cv2
import numpy as np
from slm_gsw.dcam_live_capturing import *

class beam_locator:
    def __init__(self, image, number_of_rows, number_of_cols):
        self.image = image
        self.cursor_locations = [(1000, 100), (1700, 100), (200, 200), (100, 1700)]  # Initialize cursor locations
        self.dragging = False
        self.current_cursor = None
        self.crosshair_length = 40  # Set the length of the crosshair lines
        self.rows = number_of_rows
        self.cols = number_of_cols
        # initialize all the beam corners
        self.total_num_beams = self.rows * self.cols
        self.beam_corners = np.zeros((self.total_num_beams, 2, 2), dtype=float)

    def draw_crosshair(self, img, center):
        # Draw vertical line
        cv2.line(img, (center[0], center[1] - self.crosshair_length // 2),
                 (center[0], center[1] + self.crosshair_length // 2), (0, 255, 0), 2)
        # Draw horizontal line
        cv2.line(img, (center[0] - self.crosshair_length // 2, center[1]),
                 (center[0] + self.crosshair_length // 2, center[1]), (0, 255, 0), 2)

    def display_image_with_crosshairs(self):
        cv2.namedWindow('Image with Crosshairs', cv2.WINDOW_NORMAL)  # Resizable window

        while True:
            img = self.image.copy()

            # Normalize the image to uint8 if it is uint16
            if img.dtype == np.uint16:
                img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))

            if len(img.shape) == 2:  # Check if the image is grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Draw crosshairs at cursor locations
            for loc in self.cursor_locations:
                self.draw_crosshair(img, loc)

            cv2.imshow('Image with Crosshairs', img)

            # Set mouse callback function
            cv2.setMouseCallback('Image with Crosshairs', self.mouse_callback)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if cursor is clicked
            for idx, loc in enumerate(self.cursor_locations):
                if abs(loc[0] - x) < self.crosshair_length // 2 and abs(loc[1] - y) < self.crosshair_length // 2:
                    self.dragging = True
                    self.current_cursor = idx
                    break
        elif event == cv2.EVENT_MOUSEMOVE:
            # Update cursor location if dragging
            if self.dragging:
                self.cursor_locations[self.current_cursor] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            # Stop dragging
            self.dragging = False
            self.current_cursor = None

    def get_cursor_locations(self):
        # sort the cursor locations in the following order, top left, top right, bottom left, bottom right
        self.cursor_locations = sorted(self.cursor_locations, key=lambda x: x[0])
        self.cursor_locations = sorted(self.cursor_locations, key=lambda x: x[1])

        return self.cursor_locations

    def calculate_all_beam_locations(self):
        # from the first and third elements of the cursor_locations list, we can calculate the diameter of a beam
        # the first element of the array is the top left of the beam and the third element represents the bottom right corner of the beam
        self.beam_diameter = np.sqrt((self.cursor_locations[2][0] - self.cursor_locations[0][0])**2 + (self.cursor_locations[2][1] - self.cursor_locations[0][1])**2)
        print("Beam Diameter:", self.beam_diameter)

        # from the first and second element of the list, these are the top left corners of the top left most and top right most beams
        # find the horizontal intervals of all beams
        self.horizontal_intervals = (self.cursor_locations[1][0] - self.cursor_locations[0][0]) / (self.cols - 1)
        # now to find the vertical intervals
        self.vertical_intervals = (self.cursor_locations[3][1] - self.cursor_locations[0][1]) / (self.rows - 1)

        self.horizontal_rise = (self.cursor_locations[1][1] - self.cursor_locations[0][1]) / (self.cols - 1)
        self.vertical_shift = (self.cursor_locations[3][0] - self.cursor_locations[0][0]) / (self.rows - 1)

        # now from these horizontal and vertical intervals, we can calculate the top left corners of all 25 beams and create a box using the diameter calculated as well
        # store the corners in a 25 x 4 array, for each beam, store the four corners of the box

        # Iterate over the rows
        for i in range(self.rows):
            # Iterate over the columns
            for j in range(self.cols):
                # Calculate the top left corner of the current box
                top_left = [self.cursor_locations[0][0] + j * self.horizontal_intervals + i * self.vertical_shift, self.cursor_locations[0][1] + i * self.vertical_intervals + j * self.horizontal_rise]
                
                # Calculate the bottom right corner
                bottom_right = [top_left[0] + self.beam_diameter, top_left[1] + self.beam_diameter]
                
                # Store the corners in the array
                self.beam_corners[i * self.rows + j] = [top_left, bottom_right]

# dcam_capture = DcamLiveCapturing(iDevice = 0)
# number_of_rows = 2
# number_of_columns = 2

# captured_image = dcam_capture.capture_live_images()

# # Check if an image was captured
# if captured_image is not None:
#     # # Create a resizable window
#     # cv2.namedWindow("Captured Image", cv2.WINDOW_NORMAL)

#     # # Display the captured image using OpenCV
#     # cv2.imshow("Captured Image", captured_image)
    
#     # while True:
#     #     if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Captured Image", cv2.WND_PROP_VISIBLE) < 1:
#     #         break
#     # cv2.destroyAllWindows()  # Close the window

#     locator = beam_locator(captured_image, number_of_rows, number_of_columns)

#     # Display image with crosshairs and allow user interaction
#     locator.display_image_with_crosshairs()

#     # Get cursor locations
#     cursor_locations = locator.get_cursor_locations()

#     locator.calculate_all_beam_locations()

#     beam_corners = locator.beam_corners

#     # store them in a file
#     with open("beam_corners.txt", "w") as f:
#         for loc in beam_corners:
#             top_left, bottom_right = loc
#             f.write(f"{top_left[0]}, {top_left[1]}, {bottom_right[0]}, {bottom_right[1]}\n")
# else:
#     print("No image captured.")

# def read_beam_corners(filename):
#     """
#     Reads beam corner coordinates from a file and returns them as a list of tuples.

#     Parameters:
#     filename (str): The path to the file containing the beam corners.

#     Returns:
#     beam_corners (list): A list of tuples, where each tuple contains two pairs of coordinates
#                          (top-left and bottom-right) of a beam.
#     """
#     beam_corners = []
#     with open(filename, "r") as f:
#         for line in f:
#             coordinates = list(map(float, line.strip().split(',')))
#             top_left = (coordinates[0], coordinates[1])
#             bottom_right = (coordinates[2], coordinates[3])
#             beam_corners.append((top_left, bottom_right))
#     return beam_corners

# def measure_greyscale_intensity(image, beam_corners):
#     """
#     Measures the grayscale intensity inside the rectangles described by the coordinates in beam_corners.

#     Parameters:
#     image (numpy.ndarray): The input image.
#     beam_corners (list): A list of tuples, where each tuple contains two pairs of coordinates
#                          (top-left and bottom-right) of a beam.

#     Returns:
#     intensities (list): A list of average grayscale intensities for each beam.
#     """
#     intensities = []

#     for top_left, bottom_right in beam_corners:
#         x1, y1 = map(int, top_left)
#         x2, y2 = map(int, bottom_right)
#         roi = image[y1:y2, x1:x2]
#         average_intensity = np.mean(roi)
#         intensities.append(average_intensity)

#     return intensities

# captured_image = dcam_capture.capture_single_frame()
# # Example Usage
# # Assuming `captured_image` is the image and `beam_corners_from_file` is the list of beam corners

# # Read the beam corners from the file
# beam_corners_from_file = read_beam_corners("beam_corners.txt")

# # Measure the grayscale intensity inside the rectangles described by the beam corners
# intensities = measure_greyscale_intensity(captured_image, beam_corners_from_file)

# print("Average Grayscale Intensities for Each Beam:")
# for idx, intensity in enumerate(intensities):
#     print(f"Beam {idx + 1}: {intensity}")
