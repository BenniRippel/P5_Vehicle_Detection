# Plan für train test split:
#   Shuffle KITTI daten, davon 30% als testsplit, rest mit den anderen ordnern kombinieren als training set

import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

class Classifier:
    def __init__(self):
        ### Tweak these parameters and see how the results change.
        self.color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 12  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (16, 16)  # Spatial binning dimensions
        self.hist_bins = 16  # Number of histogram bins
        self.bins_range = (0, 256)
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off
        self.y_start_stop = [360, None]  # Min and max in y to search in slide_window()

    def run(self):
        car_list, noncar_list = self.readData()
        X_train, X_test, y_train, y_test, X_Scaler = self.get_features(car_list, noncar_list)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        classifier = self.train_Classifier(X_train, y_train, X_test, y_test)

        tests = glob.glob('./test_images/*.jpg')
        for img in tests:
            test_img = cv2.imread(img)
            #t= time.time()
            self.predict_windows(test_img, classifier, X_Scaler)
            #print('Time prediction: ', time.time()-t)
        plt.show()

    def single_img_features(self, img):
        # 1) Define an empty list to receive features
        img_features = []
        # 2) Apply color conversion if other than 'BGR'
        if self.color_space != 'BGR':
            conv_to = eval('cv2.COLOR_BGR2'+self.color_space)
            feature_image = cv2.cvtColor(img, conv_to)
        else:
            feature_image = np.copy(img)
        # 3) Compute spatial features if flag is set
        if self.spatial_feat == True:
            spatial_features = self.bin_spatial(feature_image)
            # 4) Append features to list
            img_features.append(spatial_features)
        # 5) Compute histogram features if flag is set
        if self.hist_feat == True:
            hist_features = self.color_hist(feature_image)
            # 6) Append features to list
            img_features.append(hist_features)
        # 7) Compute HOG features if flag is set
        if self.hog_feat == True:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(self.get_hog_features(feature_image[:, :, channel], vis=False, feature_vec=True))
            else:
                hog_features = self.get_hog_features(feature_image[:, :, self.hog_channel], vis=False, feature_vec=True)
            # 8) Append features to list
            img_features.append(hog_features)

        # 9) Return concatenated array of features
        return np.concatenate(img_features)

    # Define a function you will pass an image
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img, windows, classifier, scaler):
        """..."""
        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        for window in windows:
            # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract features for that window using single_img_features()
            features = self.single_img_features(test_img)
            # 5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            # 6) Predict using your classifier
            prediction = classifier.predict(test_features)
            # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows

    def readData(self):
        #TODO: hier aendern für die folder, in den listen pfade angeben, dabei train test split machen
        # Read in cars and notcars
        class_1 = glob.glob('./trainingData/vehicles/vehicles/KITTI_extracted/*.png')
        class_2 = glob.glob('./trainingData/non-vehicles/non-vehicles/GTI/*.png')

        return class_1[:3900], class_2

    def get_features(self, class_1, class_2):
        """Loads images and extracts the features, splits into train and testset"""
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
        rand_state = np.random.randint(0, 100)
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

    def predict_windows(self, image, classifier, X_scaler):

        draw_image = np.copy(image)


        windows = self.slide_window(image, x_start_stop=[None, None], y_start_stop=self.y_start_stop,
                                    xy_window=(60, 60), xy_overlap=(0.5, 0.5))

        hot_windows = self.search_windows(image, windows, classifier, X_scaler)

        window_img = self.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        plt.figure()
        plt.imshow(window_img)

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

    def bin_spatial(self, img):
        """Compute binned color features (code taken from udacity class)"""
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, self.spatial_size).ravel()
        # Return the feature vector
        return features

    def color_hist(self, img):
        """Compute color histogram features (code taken from udacity class)"""
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=self.hist_bins, range=self.bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=self.hist_bins, range=self.bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=self.hist_bins, range=self.bins_range)
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

    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64),
                     xy_overlap=(0.2, 0.2)):
        """Takes an image, start and stop positions in both x and y, window size (x and y
        dimensions), and overlap fraction (for both x and y) (code taken from udacity class)"""
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        """Define a function to draw bounding boxes (code taken from udacity class)"""
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy


def main():

    Classifier().run()
# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()