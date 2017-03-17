## Vehicle Detection Project

The goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a 
 Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, 
to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and 
create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


For processing a video, the following steps are executed:

* Get filepaths for training and validation datasets (function 'read_data')
* Load, scale and shuffle the datasets (function 'get_datasets')
* Extract features from each image within the test and validation sets (function: 'extract_features')
* Train a classifier (function 'train_classifier')
* open videostream and process every frame (function 'process_frame')
    * find cars using the classifier and add the regions containing cars to the heatmap (function 'find_cars')
        * convert region of interest (roi) (bottom half of the image) to the chosen colorspace
        * for all chosen image scales:
            * scale the roi using cv2.resize
            * calculate HOG-features for the roi
            * use sliding windows to extract HOG-features and classify them to predict the presence of a car
            * if the window is classified as a car, increase the according area of the heatmap by 1
    * plot bounding boxes of cars using a thresholded integrated heatmap of the last 10 frames


[//]: # (Image References)
[image1]: ./output_images/HOG_Colorspace.png
[image2]: ./output_images/ClassifierResultsDifferentColorspaces/HLS_vs_YCrCb.png
[image3]: ./output_images/HOG_paras_orient_ppc.png
[image4]: ./output_images/Colorspaces.jpg
[image5]: ./output_images/examples.png
[image6]: ./output_images/more_examples.png
[image7]: ./output_images/show_heatmap_filter1.png
[image8]: ./output_images/show_heatmap_filter2.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The following functions are used to load images, split them in training and validation sets and extract HOG-features. 


* function 'read_data' (lines: 232-250)

    Load filepaths of image data for both classes (car and non-car), split the KITTI-Data (Cars) and Extras-Data (non-cars)
    in training and validation sets (75% / 25%). Append the GTI data to the training data. The GTI Data is not used for 
    validation due to the sequential characteristics of the data (Images are taken from a video stream), which might 
    cause overfitting. 
    
* function 'get_datasets' (lines: 252-277)

    loads the previously defined test and validation images and extracts features for each image (function 
    'extract_features') .The sklearn standard scaler is fitted to scale the training-, validation- and future test-data 
    to zero mean and unit variance. Finally, both featuresets are shuffled.
    
* function 'extract_features' (lines: 348-388)

    This is where the HOG-features (and potential other features, if the parameters are set) are extracted from the 
    training images. The contrast of every image is equalized (function 'equalize', lines 301-305) and the colorspace 
    changed to the one defined in the classes initialization function (self.color_space). The HOG-features are extracted 
    for every image channel (or one specifically defined channel), using the function 'get_hog_features' (lines 308-325).
    This function calls the skimage function 'hog', using the parameters set within the class init function. Multiple 
    features, e.g. from different color channels or additional features like color histograms, are then concatenated 
    and stored to the according featureset.
    
    
#### 2. Explain how you settled on your final choice of HOG parameters.
The chosen parameters for HOG-feature extraction were found by experiments.
The following image shows HOG Features of a car and a non-car image,  for all color channels using 'HLS' and 'YCrCb' 
colorspace. 

![alt text][image1] 


While both representations show distinct appearances for car and non-car HOG-features, we chose the 'YCrCb' 
colorspace and calculating HOG-features for all channels. The next image displays example images with windows that were classified as cars and the according heatmap for 
'HLS'-colorspace (left) and YCrCb (right). The 'HSL' based HOG features obviously create a lot of false positives and 
true positives, resulting in a lower precision and higher recall compared to 'YCrCb'. This amount of false positives is 
hard to handle, as the FP and TP areas within the heatmap often show equal values. Consequently, 'YCrCb' was chosen as 
the colorspace for HOG-features.  

![alt text][image2]

The number of orientations and pixels per cell were also found by experiment and looking at the following image. The 
number of orientations for HOG features describe the amount of bins the gradient angles get sorted into. Experiments have 
shown that the accuracy of the classifier decreases with orientations below 9, while higher numbers don't result in 
accuracy gain. Looking at the right part of the image below, the influence of the pixels per cell are obvious, less px 
per cell result in a finer grid of cells accompanied by a susceptibility to noise. Training accuracy, applying the 
classifier on test images as well as common sense show that a value of 16 px per cell is way too coarse, while a grid 
finer than 8px per cell won't result in better accuracy. Applying [Occam's Razor](https://en.wikipedia.org/wiki/Occam's_razor), 
the parameters resulting in a lower dimension feature are preferred if the accuracy of the classfier is comparable to a 
higher dimension feature. Another reason for this decision is the reduction of computation time.


![alt text][image3]

The parameter 'cells per block' refers to the block width for normalization. While this process generally improves invariance
to illumination changes, the experiments show little to no influence on the result, maybe due to the applied image 
equalization. Furthermore, the parameter 'cells per block' defines the stride of the window and consequently window 
overlap. Decreasing this value increases overlap and the total number of windows to classify, resulting in a higher 
precision but also increasing computation time. With 2 cells per step, which equal 16 px of a 64px edge and 8 px per cell,
 the overlap is 75%. Experiments show that a reduction of cells per steps reduces precision, and thus complicates false 
 positive handling.
  
 The chosen parameters are summarized in the following table:

| Parameter | Value | 
|:---|:---:|
|**Color Space**|YCrCb  | 
|**Orientations**|9  | 
|**Pixels per Cell**|8  |
|**Cells per Block**|2  | 
|**Cells per Step**|2  |
|**Image Channels**|All  |

No spatial binning or color histograms of any color channel was used. Extensive experiments have shown that the detection 
rate of either the white or the black car decreased, while false positives increased immensely, resulting in a worse 
overall performance. When plotting the pixel colors of example images in 3D-space for the color spaces 'RGB', 'HLS' and 
'YCrCb', it becomes evident that the white and black pixels of the cars cannot be isolated from background pixels 
(see following image). Relying on HOG only has several advantages. The detections don't depend on car color and thus 
decrease the influence of lighting conditions on the classifier. Furthermore, a smaller amount of features to extract 
decreases the required computational load for training and prediciton.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM using only HOG features. A grid search was conducted using 'linear' and 'rbf' kernels and varying C 
(penalty of error term) from .1 to 100. Although the 'rbf' kernel resulted in slightly higher accuracy, I chose the linear 
kernel. The rise in accuracy was not relevant (0.003%), so I chose the simpler 'linear' kernel. Varying the error penalty 
had no effect for the linear SVM.  
  

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search was implemented as proposed in the lectures ( function 'find cars', lines 134-229).

The following steps describe this process:
* initialize heatmap as a 2D array of zeros with the width and height of the frame
* get region of interest of the current frame, change colorspace if necessary
* iterate over scales
    * resize the roi dependend on the scale, new shape is w,h = (w_old/scale, h_old/scale).
    * calculate HOG features for the roi
    * calculate the positions of the upper left window corner in the dimension of HOG cells, dependend on the parameters 
    'pixels_ per_cell' (edge length of one HOG cell) and 'cells_per_step' (stride of the sliding window in HOG cells). 
    * iterate over the window positions in x- and y-direction
        * extract hog features for current window and scale using the already fitted Standard Scaler
        * predict presence of cars within the window using the classifier
        * If a car is found, increase the values of the heatmap by 1 for the window region
* append heatmap to deque for bounding box calculation


The scales were found by extensive experiments. The search in images of at least 2 different scales is necessary, as the 
cars size within the image changes with its distance toward the camera. If the search window doesn't fit to the cars 
current size within the image, false negatives might result. The computation time for each scale decreases with increasing 
scale values due to smaller images, but the number of different scale values should still be as small as possible. Searching 
the image at scales 1.0 and 1.5 turned out to be sufficient, if the overlap is at least 75%. The sliding window search 
generated too few true positives and too many false positives with less overlap, which led to problems with the heatmap 
thresholding. If more scales are used, the overlap can decrease without thresholding problems.     


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tried several different features for the classifier, including different color spaces for HOG features as well as for 
color histograms and spatial bins. Additionally, I tried to use [Udacity Data](https://github.com/udacity/self-driving-car/tree/master/annotations)
for training, but the results were quite poor as a lot of false positives occured. This may be due to the algorithm 
applied to extract negative image regions (not containing cars), which randomly chose image regions not containing cars.
Those regions often contained no texture at all and thus included very little information.
  
  
The following example images show the windows that were predicted as cars as well as the according heatmap. There are 
quite a few false positives, but the cars are detected with a high probability, leading to high values in the according 
heatmap regions (medium precision, high recall). Due to this fact, a threshold can be applied to the sum of the last 10 
heatmaps. This results in low false positives and smooth bounding boxes in the final video.

![alt text][image5]
![alt text][image6]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
The heatmap, calculated in the function 'find_cars', is appended to a deque with a max length of 10. The function 
'plot_bboxes' (lines 112-127) calculates the sum of all heatmap entries (one for every procesed frame) in this deque, 
and thresholds it. Values below 18 are set to zero, values equal or above 18 are set to 255. The function 'cv2.findContours'
is applied to this thresholded heatmap, finding all connected non-zero areas. A bounding rectangle is fit around every
area bigger than 2048px (corresponds to 0.5* window size at scale =1) using the function 'cv2.boundingRect'. The thresholds
 weres found by trial and error.
 
This approach effectively eliminates false positives, while areas containing cars easily pass the threshold due to multiple
  detections for every frame. Furthermore, all overlapping windows are combined to one bounding box.
   
The next image shows 10 consecutive frames containing windows predicted as cars, as well as the according heatmap of the frame.
![alt text][image7]
The next image displays the combined heatmap of the above frames, as well as the resulting contours and bounding boxes.
![alt text][image8]


### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This implementation works for the project video, but is quite slow (2-3 frames/s) and still contains a few false positives. 
Significant improvements in processing 
time without sacrificing accuracy will not be achieved with this approach. Calculating HOG Features for the whole region 
of interest using 2 or more scales takes too long. Furthermore, the pipeline has not been tested on other videos yet. The 
parameters were tuned for the project video, so there is no proof that it will generalize to other videos as well.

A solution to this problem using classic computer vision would be the 
following approach:
 * calculate HOG Features for the whole ROI every 10th frame, find possible cars
 * for all proposed areas, calculate [features for tracking](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack)
 * track the features to the next frame using [optical flow](http://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrl)
 * define an area around the tracked features to calculate HOG features at differen scales
 * add the windows containing predicted cars to the heatmap
 * repeat for next frame
 
 This will reduce processing time as well as false positives.
 
 Another approach that is working even better and achieving higher framerates is based on CNNs. Several pretrained models 
 exist that provide high accuracy and fast calculation while generalizing well.
