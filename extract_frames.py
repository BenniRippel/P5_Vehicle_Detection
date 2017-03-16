import numpy as np
import cv2
from moviepy.editor import VideoFileClip


class extract_frames:
    def __init__(self):
        self.num_frame=0
        self.saved=5

    def run(self):
        video='./challenge_video.mp4'
        clip = VideoFileClip(video)  # read video
        output = clip.fl_image(self.extract_frame)  # proces video; function expects color images
        output.write_videofile('nonsense.mp4', audio=False)  # write video

    def extract_frame(self, RGB_frame):
        if self.num_frame in [25, 75, 350]:
            print(self.num_frame)
            self.saved += 1
            filename = './test_images/test'+str(self.saved)+'.jpg'
            img = cv2.cvtColor(RGB_frame, cv2.COLOR_RGB2BGR)  # convert to bgr
            cv2.imwrite(filename, img)
        self.num_frame += 1
        return RGB_frame

def main():

    extract_frames().run()

# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()
