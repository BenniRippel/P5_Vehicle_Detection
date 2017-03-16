


from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from collections import deque

from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

class Classifier:

    def __init__(self):

        self.color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.color_space_feat = 'YCrCb'

        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.cells_per_step = 2  # Instead of overlap, define how many cells to step
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
        self.heat_thresh = 18

    def choose_classifier(self):
        """Search for optimum classifier"""
        car_list_train, car_list_test, noncar_list_train, noncar_list_test = self.read_data()

        X_train, X_test, y_train, y_test, self.scaler = self.get_datasets(car_list_train, car_list_test,
                                                                          noncar_list_train, noncar_list_test)
        self.get_best_classifier(X_train, y_train, X_test, y_test)

    def run_video(self, video='./challenge_video.mp4'):
        """Run the Vehicle Detection Pipeline on a input video"""

        car_list_train, car_list_test, noncar_list_train, noncar_list_test = self.read_data()
        X_train, X_test, y_train, y_test, self.scaler = self.get_datasets(car_list_train, car_list_test,
                                                                          noncar_list_train, noncar_list_test)
        self.classifier = self.train_classifier(X_train, y_train, X_test, y_test)

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
        car_list_train, car_list_test, noncar_list_train, noncar_list_test = self.read_data()
        X_train, X_test, y_train, y_test, self.scaler = self.get_datasets(car_list_train, car_list_test,
                                                                          noncar_list_train, noncar_list_test)
        self.classifier = self.train_classifier(X_train, y_train, X_test, y_test)

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

            # Define cells and steps
            nxcells = (ch1.shape[1] // self.pix_per_cell)-1  # no of cells in x-dir over image
            nycells = (ch1.shape[0] // self.pix_per_cell)-1  # no of cells in y-dir over image
            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            ncells_per_window = (window // self.pix_per_cell)-1
            nxsteps = (nxcells - ncells_per_window) // self.cells_per_step
            nysteps = (nycells - ncells_per_window) // self.cells_per_step

            # Compute individual channel HOG features for the entire image
            hog1 = self.get_hog_features(ch1, feature_vec=False)
            hog2 = self.get_hog_features(ch2, feature_vec=False)
            hog3 = self.get_hog_features(ch3, feature_vec=False)

            for xb in range(nxsteps): # indices in hog cells
                for yb in range(nysteps):
                    ypos = yb * self.cells_per_step
                    xpos = xb * self.cells_per_step
                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos + ncells_per_window, xpos:xpos + ncells_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos + ncells_per_window, xpos:xpos + ncells_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos + ncells_per_window, xpos:xpos + ncells_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                    # index in pixelspace
                    xleft = xpos * self.pix_per_cell
                    ytop = ypos * self.pix_per_cell

                    # Get color features
                    if self.spatial_feat or self.hist_feat:
                        # Extract the image patch
                        subimg = cv2.resize(Color_feat_img[ytop:ytop + window, xleft:xleft + window], (64, 64))
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


    def read_data(self):
        """take 25% of the kitti data and 25% of the non-vehicles/Extras for testing, add rest to training data"""
        # Read in cars and notcars
        kitti = glob.glob('./trainingData/vehicles/vehicles/KITTI_extracted/*.png')
        length_kitti = len(kitti)
        class_1_test = kitti[int(0.75*length_kitti):]
        class_1_train = kitti[:int(0.75*length_kitti)]
        class_1_train.extend(glob.glob('./trainingData/vehicles/vehicles/GTI_Far/*.png'))
        class_1_train.extend(glob.glob('./trainingData/vehicles/vehicles/GTI_Left/*.png'))
        class_1_train.extend(glob.glob('./trainingData/vehicles/vehicles/GTI_MiddleClose/*.png'))
        class_1_train.extend(glob.glob('./trainingData/vehicles/vehicles/GTI_Right/*.png'))

        extras = glob.glob('./trainingData/non-vehicles/non-vehicles/Extras/*.png')
        length_extras = len(extras)
        class_2_test = extras[int(0.75*length_kitti):]
        class_2_train = extras[:int(0.75*length_kitti)]
        class_2_train.extend(glob.glob('./trainingData/non-vehicles/non-vehicles/GTI/*.png'))

        return class_1_train, class_1_test, class_2_train, class_2_test

    def get_datasets(self, class_1_train, class_1_test, class_2_train, class_2_test):
        """Loads images and extracts the features, splits into train and testset"""
        # extract positive and negative features and stack them
        class_1_feats_train = self.extract_features(class_1_train)
        class_2_feats_train = self.extract_features(class_2_train)
        X_train = np.vstack((class_1_feats_train, class_2_feats_train)).astype(np.float64)
        class_1_feats_test = self.extract_features(class_1_test)
        class_2_feats_test = self.extract_features(class_2_test)
        X_test = np.vstack((class_1_feats_test, class_2_feats_test)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        scaled_X_train = X_scaler.transform(X_train)
        scaled_X_test = X_scaler.transform(X_test)

        # Define the labels vector
        y_train = np.hstack((np.ones(len(class_1_feats_train)), np.zeros(len(class_2_feats_train))))
        y_test = np.hstack((np.ones(len(class_1_feats_test)), np.zeros(len(class_2_feats_test))))

        # Shuffle both sets
        idx_train = np.arange(len(y_train))
        np.random.shuffle(idx_train)
        idx_test = np.arange(len(y_test))
        np.random.shuffle(idx_test)

        return scaled_X_train[idx_train], scaled_X_test[idx_test], y_train[idx_train], y_test[idx_test], X_scaler

    def train_classifier(self, X_train, y_train, X_test, y_test):
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

    def get_best_classifier(self, X_train, y_train, X_test, y_test):
        """Grid search classifier for best result"""
        for kernel in ['rbf', 'linear']:
            for c in [0.1, 10]:
                svm = SVC(kernel=kernel, C=c)
                svm.fit(X_train, y_train)
                score=svm.score(X_test, y_test)
                print('Kernel: ', kernel, ' C: ', c, ' Score: ', score)

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

    def example_plots_HOG(self, colorspace='YCrCb'):
        """Create example images of HOG Features for cars and noncars for all relevant channels"""
        # load random images
        car = glob.glob('./trainingData/vehicles/vehicles/KITTI_extracted/*.png')
        noncar =glob.glob('./trainingData/non-vehicles/non-vehicles/Extras/*.png')
        # rnd_idx_car = np.random.randint(len(car)-1)
        # rnd_idx_noncar = np.random.randint(len(noncar)-1)
        rnd_idx_car = 100
        rnd_idx_noncar = 24

        car_img = cv2.imread(car[rnd_idx_car])
        noncar_img = cv2.imread(noncar[rnd_idx_noncar])
        # convert to colorspace
        conv=eval('cv2.COLOR_BGR2'+colorspace)
        car_cs = cv2.cvtColor(car_img, conv)
        noncar_cs = cv2.cvtColor(noncar_img, conv)

        # plot
        fig=plt.figure()
        fig.suptitle('HOG Features of the colorspace '+colorspace, fontsize=16)
        #rgb car
        fig.add_subplot(4, 4, 1)
        plt.imshow(cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB))
        plt.title('Car RBG', fontsize=8)
        plt.axis('off')

        #colorspace car
        fig.add_subplot(4, 4, 2)
        plt.imshow(car_cs)
        plt.title('Car '+colorspace, fontsize=8)
        plt.axis('off')

        #rgb noncar
        fig.add_subplot(4, 4, 3)
        plt.imshow(cv2.cvtColor(noncar_img, cv2.COLOR_BGR2RGB))
        plt.title('Non-Car RBG', fontsize=8)
        plt.axis('off')

        #colorspace noncar
        fig.add_subplot(4, 4, 4)
        plt.imshow(noncar_cs)
        plt.title('Non-Car '+colorspace, fontsize=8)
        plt.axis('off')

        # colorspace car channel 0
        fig.add_subplot(4,4,5)
        plt.imshow(car_cs[:,:,0], cmap='gray')
        plt.title('Car'+colorspace+' chan 0', fontsize=8)
        plt.axis('off')

        # colorspace car channel 0 HOG Feats
        fig.add_subplot(4,4,6)
        plt.imshow( hog(car_cs[:,:,0], orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block),
                                      transform_sqrt=True,
                                      visualise=True, feature_vector=False)[1] ,cmap='gray')
        plt.title('Car'+colorspace+' chan 0 HOG', fontsize=8)
        plt.axis('off')

        # colorspace noncar channel 0
        fig.add_subplot(4,4,7)
        plt.imshow(noncar_cs[:,:,0], cmap='gray')
        plt.title('Non-Car'+colorspace+' chan 0', fontsize=8)
        plt.axis('off')

        # colorspace noncar channel 0 HOG Feats
        fig.add_subplot(4,4,8)
        plt.imshow( hog(noncar_cs[:,:,0], orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block),
                                      transform_sqrt=True,
                                      visualise=True, feature_vector=False)[1] ,cmap='gray')
        plt.title('Non-Car'+colorspace+' chan 0 HOG', fontsize=8)
        plt.axis('off')

        # colorspace car channel 1
        fig.add_subplot(4,4,9)
        plt.imshow(car_cs[:,:,1], cmap='gray')
        plt.title('Car'+colorspace+' chan 1', fontsize=8)
        plt.axis('off')

        # colorspace car channel 1 HOG Feats
        fig.add_subplot(4,4,10)
        plt.imshow( hog(car_cs[:,:,1], orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block),
                                      transform_sqrt=True,
                                      visualise=True, feature_vector=False)[1] ,cmap='gray')
        plt.title('Car'+colorspace+' chan 1 HOG', fontsize=8)
        plt.axis('off')

        # colorspace noncar channel 1
        fig.add_subplot(4,4,11)
        plt.imshow(noncar_cs[:,:,1], cmap='gray')
        plt.title('Non-Car'+colorspace+' chan 1', fontsize=8)
        plt.axis('off')

        # colorspace noncar channel 1 HOG Feats
        fig.add_subplot(4,4,12)
        plt.imshow( hog(noncar_cs[:,:,1], orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block),
                                      transform_sqrt=True,
                                      visualise=True, feature_vector=False)[1] ,cmap='gray')
        plt.title('Non-Car'+colorspace+' chan 1 HOG', fontsize=8)
        plt.axis('off')

        # colorspace car channel 2
        fig.add_subplot(4,4,13)
        plt.imshow(car_cs[:,:,2], cmap='gray')
        plt.title('Car'+colorspace+' chan 2', fontsize=8)
        plt.axis('off')

        # colorspace car channel 2 HOG Feats
        fig.add_subplot(4,4,14)
        plt.imshow( hog(car_cs[:,:,2], orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block),
                                      transform_sqrt=True,
                                      visualise=True, feature_vector=False)[1] ,cmap='gray')
        plt.title('Car'+colorspace+' chan 2 HOG', fontsize=8)
        plt.axis('off')

        # colorspace noncar channel 2
        fig.add_subplot(4,4,15)
        plt.imshow(noncar_cs[:,:,2], cmap='gray')
        plt.title('Non-Car'+colorspace+' chan 2', fontsize=8)
        plt.axis('off')

        # colorspace noncar channel 2 HOG Feats
        fig.add_subplot(4,4,16)
        plt.imshow( hog(noncar_cs[:,:,2], orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block),
                                      transform_sqrt=True,
                                      visualise=True, feature_vector=False)[1] ,cmap='gray')
        plt.title('Non-Car'+colorspace+' chan 2 HOG', fontsize=8)
        plt.axis('off')

        plt.savefig('./output_images/HOG_Features_'+colorspace+'_.png', dpi=400)
        plt.show()

    def example_HOG_paras(self):
        """Plots for explaining HOG Parameters"""
        car = glob.glob('./trainingData/vehicles/vehicles/KITTI_extracted/*.png')
        rnd_idx_car = np.random.randint(len(car)-1)
        car_rgb = cv2.cvtColor(cv2.imread(car[rnd_idx_car]), cv2.COLOR_BGR2RGB)
        car_img = cv2.cvtColor(cv2.imread(car[rnd_idx_car]), cv2.COLOR_BGR2YCrCb)[:,:,0]

        # vary pixels per cell
        fig=plt.figure()
        fig.suptitle('HOG Features: varying pixels per cell', fontsize=16)
        fig.add_subplot(1, 4, 1)
        plt.imshow(car_rgb)
        plt.title('RGB', fontsize=10)
        plt.axis('off')

        for idx, ppc in enumerate([4, 8, 16]):
            fig.add_subplot(1, 4, 2+idx)
            plt.imshow(hog(car_img[:, :], orientations=self.orient, pixels_per_cell=(ppc, ppc),
                           cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=True,
                           visualise=True, feature_vector=False)[1], cmap = 'gray')
            plt.title('Px per cell = '+str(ppc), fontsize=10)
            plt.axis('off')
        plt.savefig('./output_images/HOG_Features_ppc.png', dpi=400)

        # vary orientations
        fig=plt.figure()
        fig.suptitle('HOG Features: varying orientations', fontsize=16)
        fig.add_subplot(1, 4, 1)
        plt.imshow(car_rgb)
        plt.title('RGB', fontsize=10)
        plt.axis('off')

        for idx, ori in enumerate([6, 9, 12]):
            fig.add_subplot(1, 4, 2+idx)
            plt.imshow(hog(car_img[:, :], orientations=ori, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=True,
                           visualise=True, feature_vector=False)[1], cmap = 'gray')
            plt.title('orientations = '+str(ori), fontsize=10)
            plt.axis('off')
        plt.savefig('./output_images/HOG_Features_orient.png', dpi=400)
        plt.show()

    def show_heatmap_filter(self):
        """Create plots for explaining the heatmap filter"""
        car_list_train, car_list_test, noncar_list_train, noncar_list_test = self.read_data()
        X_train, X_test, y_train, y_test, self.scaler = self.get_datasets(car_list_train, car_list_test,
                                                                          noncar_list_train, noncar_list_test)
        self.classifier = self.train_classifier(X_train, y_train, X_test, y_test)

        # plot single frames with windows and heatmap
        tests = glob.glob('./test_images/showHeatmap*.png')
        fig = plt.figure()
        idx = 1
        self.heatmap = []
        for img in tests:
            test_img = cv2.imread(img)
            t= time.time()
            img=self.find_cars(test_img, return_image=True)
            print('Time prediction: ', time.time()-t)
            fig.add_subplot(len(tests)/2, 4, idx)
            idx+=1
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            fig.add_subplot(len(tests)/2, 4, idx)
            idx+=1
            plt.imshow(self.heatmap[-1], cmap='hot')
            plt.axis('off')
        plt.subplots_adjust(left=0.28, bottom=0.01, right=0.69, top=0.98, wspace=0.03, hspace=0.03)
        plt.savefig('./output_images/show_heatmap_filter1.png', dpi=400)

        #plot frame with resulting bounding boxes
        fig2 = plt.figure()
        fig2.add_subplot(1, 3, 1)
        img = cv2.imread(tests[-1])
        plt.imshow(self.plot_bboxes(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        plt.axis('off')
        plt.title('Bounding Boxes')
        # plot resulting heatmap
        heatmap = np.zeros_like(img[:, :, 0])
        for a in self.heatmap:
            heatmap += a
        img2 = heatmap / np.max(heatmap)
        fig2.add_subplot(1, 3, 2)
        plt.imshow(img2, cmap='hot')
        plt.axis('off')
        plt.title('Combined Heatmap')
        # plot found contours
        # threshold heatmap
        heatmap[heatmap<self.heat_thresh]=0
        heatmap[heatmap>=self.heat_thresh] = 255
        img3=np.zeros_like(img)
        # find contours and plot their bounding rectangles
        im2, contours, hierarchy = cv2.findContours(heatmap.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt)>2048:
                cv2.drawContours(img3, [cnt], 0, (0, 255, 0), -1)

        fig2.add_subplot(1, 3, 3)
        plt.imshow(img3, cmap='gray')
        plt.axis('off')
        plt.title('Contours')
        plt.savefig('./output_images/show_heatmap_filter2.png', dpi=400)
        plt.show()



def main():
    #Classifier().choose_classifier()
    # Classifier().run_video()
    # Classifier().run_images()
    Classifier().show_heatmap_filter()
# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()