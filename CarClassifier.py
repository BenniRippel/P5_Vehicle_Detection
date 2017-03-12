# Plan für train test split:
#   Shuffle KITTI daten, davon 30% als testsplit, rest mit den anderen ordnern kombinieren als training set
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from collections import deque
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split


class Classifier:
    def __init__(self):
        self.color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.color_space_feat = 'YCrCb'
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32)  # Spatial binning dimensions
        self.hist_bins = 64  # Number of histogram bins
        self.spatial_feat = False  # Spatial features on or off
        self.hist_feat = False  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off
        self.y_start_stop = [400, 656]  # Min and max in y to search in slide_window()
        self.scale = [1.0, 1.5] # max 1.5

        self.scaler = []
        self.classifier = []

        self.heatmap = deque(maxlen=10)
        self.heat_thresh = 17 #17

    def run_video(self, video='./project_video.mp4'):
        """Run the Vehicle Detection Pipeline on a input video"""
        car_list, noncar_list = self.readData()
        X_train, X_test, y_train, y_test, self.scaler = self.get_features(car_list, noncar_list)
        self.classifier = self.train_Classifier(X_train, y_train, X_test, y_test)

        out_file = video[:-4] + '_output.mp4'  # output file
        clip = VideoFileClip(video)  # read video
        output = clip.fl_image(self.process_frame)  # process video; function expects color images
        output.write_videofile(out_file, audio=False)  # write video

    def process_frame(self, img):
        """processes a videoframe, after a classifier was trained"""
        self.find_cars(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), return_image=True)
        img = self.plot_bboxes(img)
        return self.add_heatmap(img)

    def add_heatmap(self, frame):
        """Add the img to the upper right corner of the frame, scale by 1/3"""
        h, w, c = frame.shape
        h_n = int(h / 3)
        w_n = int(w / 3)
        heatmap = np.zeros_like(frame[:,:,0])
        for a in self.heatmap:
            heatmap +=a

        img = heatmap/np.max(heatmap)
        img = np.dstack((img * 255, img * 255, img * 255))
        img = cv2.resize(img, (w_n, h_n), interpolation=cv2.INTER_AREA)
        frame[:h_n, -w_n:, :] = img
        return frame

    def run_images(self):
        """Run on example images"""
        car_list, noncar_list = self.readData()
        X_train, X_test, y_train, y_test, self.scaler = self.get_features(car_list, noncar_list)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        print(np.max(X_train))
        self.classifier = self.train_Classifier(X_train, y_train, X_test, y_test)

        tests = glob.glob('./test_images/vlc*.png')
        fig = plt.figure()
        idx = 1

        for img in tests:
            test_img = cv2.imread(img)
            t= time.time()
            self.heatmap = []
            img=self.find_cars(test_img, return_image=True)
            print('Time prediction: ', time.time()-t)
            fig.add_subplot(len(tests), 2, idx)
            idx+=1
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            fig.add_subplot(len(tests), 2, idx)
            idx+=1
            plt.imshow(self.heatmap[-1], cmap='hot')
            plt.axis('off')
        plt.subplots_adjust(left=0.28, bottom=0.01, right=0.69, top=0.98, wspace=0.03, hspace=0.03)
        plt.savefig('./output_images/test_heatmap.png', dpi=400)
        plt.show()

    def plot_bboxes(self, img):
        """Plot areas where the heatmap surpasses a certain threshold"""
        # add all heatmaps in deque
        heatmap = np.zeros_like(img[:,:,0])
        for a in self.heatmap:
            heatmap +=a
        # threshold heatmap
        heatmap[heatmap<self.heat_thresh]=0
        heatmap[heatmap>=self.heat_thresh] = 255
        # find contours and plot their bounding rectangles
        im2, contours, hierarchy = cv2.findContours(heatmap.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt)>2048:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 6)
        return img

    def find_cars(self, img, return_image=False):
        """Find cars in an img using self.classifier and a sliding window approach. Add locations to heatmap and draw
        the positive windows if return_image = True. Most of the code taken from udacity class """
        #init image to draw on
        if return_image: draw_img = np.copy(img)
        #init heat
        heat = np.zeros_like(draw_img[:, :, 0])

        # check for valid y_start_stop
        if self.y_start_stop[0] == None: self.y_start_stop[0]=img.shape[0]/2
        if self.y_start_stop[1] == None: self.y_start_stop[1]=img.shape[0]-1
        # get image slice to get HOG for
        img_tosearch = img[self.y_start_stop[0]:self.y_start_stop[1], :, :]

        # convert colorspace for HOG features
        if self.color_space is not 'BGR':
            conv_to = eval('cv2.COLOR_BGR2'+self.color_space)
            ctransHOG = cv2.cvtColor(img_tosearch, conv_to)
        else:
            ctransHOG = img_tosearch
        # convert colorspace for color features
        if self.color_space_feat is not 'BGR':
            conv_to = eval('cv2.COLOR_BGR2'+self.color_space_feat)
            ctransCOL = cv2.cvtColor(img_tosearch, conv_to)
        else:
            ctransCOL = img_tosearch

        for scale in self.scale:
            #scale
            if scale != 1:
                imshape = ctransHOG.shape
                HOG_color_img = cv2.resize(ctransHOG, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
                Color_feat_img = cv2.resize(ctransCOL, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
            else:
                HOG_color_img = ctransHOG
                Color_feat_img = ctransCOL
            # colorchannels
            ch1 = HOG_color_img[:, :, 0]
            ch2 = HOG_color_img[:, :, 1]
            ch3 = HOG_color_img[:, :, 2]

            # Define blocks and steps
            nxblocks = (ch1.shape[1] // self.pix_per_cell) - 1 # no of blocks in x-dir over image
            nyblocks = (ch1.shape[0] // self.pix_per_cell) - 1 # no of blocks in y-dir over image
            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            nblocks_per_window = (window // self.pix_per_cell) - 1
            cells_per_step = 2  # Instead of overlap, define how many cells to step
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step

            # Compute individual channel HOG features for the entire image
            hog1 = self.get_hog_features(ch1, feature_vec=False)
            hog2 = self.get_hog_features(ch2, feature_vec=False)
            hog3 = self.get_hog_features(ch3, feature_vec=False)

            for xb in range(nxsteps): # indices in hog cells
                for yb in range(nysteps):
                    ypos = yb * cells_per_step
                    xpos = xb * cells_per_step
                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                    # index in pixelspace
                    xleft = xpos * self.pix_per_cell
                    ytop = ypos * self.pix_per_cell

                    # Extract the image patch
                    subimg = cv2.resize(Color_feat_img[ytop:ytop + window, xleft:xleft + window], (64, 64))

                    # Get color features
                    if self.spatial_feat:
                        spatial_features = self.bin_spatial(subimg)
                    else:
                        spatial_features = []
                    if self.hist_feat:
                        hist_features = self.color_hist(subimg)
                    else:
                        hist_features = []
                    # Scale features and make a prediction
                    test_features =self.scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                    test_prediction = self.classifier.predict(test_features)

                    if test_prediction == 1:
                        xbox_left = np.int(xleft * scale)
                        ytop_draw = np.int(ytop * scale)
                        win_draw = np.int(window * scale)
                        if return_image:
                            cv2.rectangle(draw_img, (xbox_left, ytop_draw + self.y_start_stop[0]),
                                          (xbox_left + win_draw, ytop_draw + win_draw + self.y_start_stop[0]), (0, 0, 255), 6)
                        heat[ytop_draw+self.y_start_stop[0]:ytop_draw+win_draw+self.y_start_stop[0], xbox_left:xbox_left+win_draw] += 1
        self.heatmap.append(heat)
        if return_image:
            return draw_img


    def readData(self):
        #TODO: hier aendern für die folder, in den listen pfade angeben, dabei train test split machen
        # Read in cars and notcars
        class_1 = glob.glob('./trainingData/vehicles/vehicles/KITTI_extracted/*.png')
        class_1.extend(glob.glob('./trainingData/vehicles/vehicles/GTI_Far/*.png'))
        class_1.extend(glob.glob('./trainingData/vehicles/vehicles/GTI_Left/*.png'))
        class_1.extend(glob.glob('./trainingData/vehicles/vehicles/GTI_MiddleClose/*.png'))
        class_1.extend(glob.glob('./trainingData/vehicles/vehicles/GTI_Right/*.png'))
        #class_1.extend(glob.glob('./trainingData/vehicles/vehicles/UdacityCar/*.png'))

        class_2 = glob.glob('./trainingData/non-vehicles/non-vehicles/Extras/*.png')
        class_2.extend(glob.glob('./trainingData/non-vehicles/non-vehicles/GTI/*.png'))
        #class_2.extend(glob.glob('./trainingData/non-vehicles/non-vehicles/UdacityNon_Car/*.png'))

        # # Udacity and KITTI
        #class_1=glob.glob('./trainingData/vehicles/vehicles/UdacityCar/*.png')
        #class_1.extend(glob.glob('./trainingData/vehicles/vehicles/KITTI_extracted/*.png'))
        #class_2=glob.glob('./trainingData/non-vehicles/non-vehicles/UdacityNon_Car/*.png')
        #class_2.extend(glob.glob('./trainingData/non-vehicles/non-vehicles/Extras/*.png'))
        return class_1, class_2

    def get_features(self, class_1, class_2):
        """Loads images and extracts the features, splits into train and testset"""
        # extract positive and negative features and stack them
        class_1_feats = self.extract_features(class_1)
        class_2_feats = self.extract_features(class_2)
        X = np.vstack((class_1_feats, class_2_feats)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(class_1_feats)), np.zeros(len(class_2_feats))))

        # Split up data into randomized training and test sets
        rand_state = 43
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
        return X_train, X_test, y_train, y_test, X_scaler

    def train_Classifier(self, X_train, y_train, X_test, y_test):
        """Trains a Classifier"""
        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        return svc

    def equalize(self, img):
        """equalize BGR Image"""
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    ###########################################
    def get_hog_features(self, img, vis=False, feature_vec=True):
        """Return HOG features and visualization (code taken from udacity class)"""
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block),
                                      transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=self.orient,
                          pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                          cells_per_block=(self.cell_per_block, self.cell_per_block),
                          transform_sqrt=True,
                          visualise=vis, feature_vector=feature_vec)
            return features

    def bin_spatial(self, img, ravel=True):
        """Compute binned color features (code taken from udacity class)"""
        # Use cv2.resize().ravel() to create the feature vector
        if ravel:
            features = cv2.resize(img, self.spatial_size).ravel()
        else:
            features = cv2.resize(img, self.spatial_size)
        # Return the feature vector
        return features

    def color_hist(self, img):
        """Compute color histogram features (code taken from udacity class)"""
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=self.hist_bins)
        channel2_hist = np.histogram(img[:, :, 1], bins=self.hist_bins)
        channel3_hist = np.histogram(img[:, :, 2], bins=self.hist_bins)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def extract_features(self, imgs):
        """Extract features from a list of images. Have this function call bin_spatial() and
        color_hist() (code taken from udacity class)"""
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            file_features = []
            # Read in each one by one
            image = cv2.imread(file)
            # equalize histogram
            image = self.equalize(image)

            # apply color conversion if other than 'BGR'
            if self.color_space != 'BGR':
                    conv_to = eval('cv2.COLOR_BGR2'+self.color_space)
                    feature_image = cv2.cvtColor(image, conv_to)
            else:
                feature_image = np.copy(image)

            if self.spatial_feat == True:
                spatial_features = self.bin_spatial(feature_image)
                file_features.append(spatial_features)
            if self.hist_feat == True:
                # Apply color_hist()
                hist_features = self.color_hist(feature_image)
                file_features.append(hist_features)
            if self.hog_feat == True:
                # Call get_hog_features() with vis=False, feature_vec=True
                if self.hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(self.get_hog_features(feature_image[:, :, channel], vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)
                else:
                    hog_features = self.get_hog_features(feature_image[:, :, self.hog_channel], vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        # Return list of feature vectors
        return features


def main():

    Classifier().run_video()
    #Classifier().run_images()
# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()