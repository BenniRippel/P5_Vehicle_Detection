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
from scipy.ndimage.measurements import label

class Classifier:
    def __init__(self):
        ### Tweak these parameters and see how the results change.
        self.color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (16, 16)  # Spatial binning dimensions
        self.hist_bins = 16  # Number of histogram bins
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off
        self.y_start_stop = [400, 656]  # Min and max in y to search in slide_window()
        self.scale = [1, 1.5]

        self.scaler = []
        self.classifier = []

        self.heatmap = deque(maxlen=5)
        self.heat_thresh = 2

    def run_video(self, video='./test_video.mp4'):
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
        img = np.mean(self.heatmap, axis=0)/np.max(np.mean(self.heatmap, axis=0))
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

        tests = glob.glob('./test_images/*.jpg')
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
        """Plot areas wihere the heatmap surpasses a certain threshold"""
        # get mean of last heatmaps
        heatmap = np.mean(self.heatmap, axis=0)
        # threshold heatmap
        heatmap[heatmap<self.heat_thresh]=0
        # get labeled heatmap
        labeled, numCars = label(heatmap)
        #draw cars in  image
        for car in range(1, numCars+1):
            # Find pixels with each car_number label value
            nonzero = (labeled == car).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
            # Return the image
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

        # convert colorspace
        if self.color_space is not 'BGR':
            conv_to = eval('cv2.COLOR_BGR2'+self.color_space)
            ctrans = cv2.cvtColor(img_tosearch, conv_to)
        else:
            ctrans = img_tosearch

        for scale in self.scale:
            #scale
            if scale != 1:
                imshape = ctrans.shape
                ctrans_tosearch = cv2.resize(ctrans, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
            else:
                ctrans_tosearch = ctrans
            # colorchannels
            ch1 = ctrans_tosearch[:, :, 0]
            ch2 = ctrans_tosearch[:, :, 1]
            ch3 = ctrans_tosearch[:, :, 2]

            # Define blocks and steps as above
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
                    subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                    # Get color features
                    spatial_features = self.bin_spatial(subimg)
                    hist_features = self.color_hist(subimg)

                    # Scale features and make a prediction
                    test_features = self.scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
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

    def readData(self):
        #TODO: hier aendern für die folder, in den listen pfade angeben, dabei train test split machen
        # Read in cars and notcars
        class_1 = glob.glob('./trainingData/vehicles/vehicles/KITTI_extracted/*.png')
        #class_1.extend(glob.glob('./trainingData/vehicles/vehicles/GTI_Far/*.png'))
        #class_1.extend(glob.glob('./trainingData/vehicles/vehicles/GTI_Left/*.png'))
        #class_1.extend(glob.glob('./trainingData/vehicles/vehicles/GTI_MiddleClose/*.png'))
        #class_1.extend(glob.glob('./trainingData/vehicles/vehicles/GTI_Right/*.png'))
        class_1.extend(glob.glob('./trainingData/vehicles/vehicles/UdacityCar/*.png'))

        class_2=glob.glob('./trainingData/non-vehicles/non-vehicles/Extras/*.png')
        #class_2.extend(glob.glob('./trainingData/non-vehicles/non-vehicles/GTI/*.png'))
        class_2.extend(glob.glob('./trainingData/non-vehicles/non-vehicles/UdacityNon_Car/*.png'))

        #Udacity and KITTI
        # class_1=glob.glob('./trainingData/vehicles/vehicles/UdacityCar/*.png')
        # class_1.extend(glob.glob('./trainingData/vehicles/vehicles/KITTI_extracted/*.png'))
        # class_2=glob.glob('./trainingData/non-vehicles/non-vehicles/UdacityNon_Car/*.png')
        # class_2.extend(glob.glob('./trainingData/non-vehicles/non-vehicles/Extras/*.png'))
        return class_1, class_2

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

    Classifier().run_video()
    #Classifier().run_images()
# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()