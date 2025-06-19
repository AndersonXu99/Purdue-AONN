"""
Sample script for showing captured image with dcam.py.
This script recognizes the camera and acquires with dcam.py.
The acquired images are displayed live with OpenCV.

This sample source code just shows how to use DCAM-API.
The performance is not guaranteed.

Original script date: 2021-06-18
Original copyright: Copyright (C) 2021-2024 Hamamatsu Photonics K.K.

Modified by: Anderson Xu
Date: 2024-06-02
"""

__date__ = '2021-06-18'
__copyright__ = 'Copyright (C) 2021-2024 Hamamatsu Photonics K.K.'

import cv2
from slm_gsw.dcam import *
import numpy as np

class DcamLiveCapturing:
    """
    A class used to capture live images from a DCAM device.

    Attributes
    ----------
    iDevice : int
        The device index.
    dcam : Dcam
        An instance of the Dcam class.

    Methods
    -------
    __init__(iDevice=0):
        Initializes the DcamLiveCapturing class with the specified device index.
    dcamtest_show_framedata(data, windowtitle, windowstatus):
        Displays the captured image data in a window.
    dcamtest_thread_live():
        Captures and displays live images until the window is closed or 'q' is pressed.
        If 'c' is pressed, the current image is saved as 'captured_image.png' and returned as a numpy array.
    capture_live_images():
        Initializes the DCAM API, opens the device, allocates the buffer, captures live images, and then cleans up.
    capture_single_frame():
        Initializes the DCAM API, opens the device, allocates the buffer, captures a single frame, and then cleans up.
    """
    
    def __init__(self, iDevice=0):
        self.iDevice = iDevice
        self.dcam = None

    def dcamtest_show_framedata(self, data, windowtitle, windowstatus):
        """Show image data.
        Show numpy buffer as an image with OpenCV function.

        Args:
            data (void): NumPy array.
            windowtitle (char): Window name.
            windowstatus (int): Last window status returned by dcamtest_show_framedata function. Specify 0 when calling the first time.

        Returns:
            int: Window status.
        """
        if windowstatus > 0 and cv2.getWindowProperty(windowtitle, cv2.WND_PROP_VISIBLE) == 0:
            return -1
            # Window has been closed.
        if windowstatus < 0:
            return -1
            # Window is already closed.

        if data.dtype == np.uint16:
            imax = np.amax(data)
            if imax > 0:
                imul = int(65535 / imax)
                data = data * imul
            cv2.namedWindow(windowtitle, cv2.WINDOW_NORMAL)
            cv2.imshow(windowtitle, data)
            return 1
        else:
            print('-NG: dcamtest_show_image(data) only support Numpy.uint16 data')
            return -1

    def dcamtest_thread_live(self):
        """Show live images.
        Capture and show live images.

        Returns:
            Captured image as numpy array, or None if no image is captured.
        """
        if self.dcam.cap_start() is not False:
            timeout_milisec = 100
            iWindowStatus = 0
            captured_image = None
            
            while iWindowStatus >= 0:
                if self.dcam.wait_capevent_frameready(timeout_milisec) is not False:
                    data = self.dcam.buf_getlastframedata()
                    iWindowStatus = self.dcamtest_show_framedata(data, 'test', iWindowStatus)
                else:
                    dcamerr = self.dcam.lasterr()
                    if dcamerr.is_timeout():
                        print('===: timeout')
                    else:
                        print('-NG: Dcam.wait_event() fails with error {}'.format(dcamerr))
                        break

                key = cv2.waitKey(1)
                if key == ord('q') or key == ord('Q'):  # if 'q' was pressed with the live window, close it
                    break
                elif key == ord('c') or key == ord('C'):
                    if data.dtype == np.uint16:
                        imax = np.amax(data)
                        if imax > 0:
                            imul = int(65535 / imax)
                            data = data * imul
                        filename = 'captured_image.png'
                        cv2.imwrite(filename, data)
                        print(f'Image captured and saved as {filename}')
                        captured_image = data  # Save the image as a numpy array
                        iWindowStatus = -1  # Set to -1 to close the window
                        cv2.destroyWindow('test')

            self.dcam.cap_stop()
            return captured_image
        else:
            print('-NG: Dcam.cap_start() fails with error {}'.format(self.dcam.lasterr()))
            return None

    def capture_live_images(self):
        """Capture live images.
        Recognize camera and capture live images.

        Returns:
            Captured image as numpy array, or None if no image is captured.
        """
        if Dcamapi.init() is not False:
            self.dcam = Dcam(self.iDevice)
            if self.dcam.dev_open() is not False:
                if self.dcam.buf_alloc(3) is not False:
                    captured_image = self.dcamtest_thread_live()
                    self.dcam.buf_release()
                    self.dcam.dev_close()
                    Dcamapi.uninit()
                    return captured_image
                else:
                    print('-NG: Dcam.buf_alloc(3) fails with error {}'.format(self.dcam.lasterr()))
                self.dcam.dev_close()
            else:
                print('-NG: Dcam.dev_open() fails with error {}'.format(self.dcam.lasterr()))
        else:
            print('-NG: Dcamapi.init() fails with error {}'.format(Dcamapi.lasterr()))

        Dcamapi.uninit()
        return None
    
    def capture_single_frame (self):
        if Dcamapi.init() is not False:
            self.dcam = Dcam(self.iDevice)
            if self.dcam.dev_open() is not False:
                if self.dcam.buf_alloc(3) is not False:
                    if self.dcam.cap_start() is not False:
                        timeout_milisec = 1000
                        if self.dcam.wait_capevent_frameready(timeout_milisec) is not False:
                            data = self.dcam.buf_getlastframedata()
                            self.dcam.cap_stop()
                            self.dcam.buf_release()
                            self.dcam.dev_close()
                            Dcamapi.uninit()
                            return data
                        else:
                            print('-NG: Dcam.wait_event() fails with error {}'.format(self.dcam.lasterr()))
                    else:
                        print('-NG: Dcam.cap_start() fails with error {}'.format(self.dcam.lasterr()))
                    self.dcam.buf_release()
                else:
                    print('-NG: Dcam.buf_alloc(3) fails with error {}'.format(self.dcam.lasterr()))
                self.dcam.dev_close()
            else:
                print('-NG: Dcam.dev_open() fails with error {}'.format(self.dcam.lasterr()))
        else:
            print('-NG: Dcamapi.init() fails with error {}'.format(Dcamapi.lasterr()))

        Dcamapi.uninit()
        return None

def dcam_live_capturing(iDevice=0):
    """Wrapper function for backward compatibility."""
    dcam_capture = DcamLiveCapturing(iDevice)
    return dcam_capture.capture_live_images()


if __name__ == '__main__':
    dcam_live_capturing()