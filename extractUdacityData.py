import pandas as pd
import numpy as np
import cv2
import csv
import os
import matplotlib.pyplot as plt


class extractData:

    def __init__(self, dataFolder='./UdacityData/', img_size=(64, 64, 3)):
        self.dataFolder = dataFolder
        self.data = None
        self.img_size= img_size

    def run(self):
        self.read_labels()
        self.get_labeled_images('Car')


    def read_labels(self):
        file='./UdacityData/labels.csv'
        with open(file, mode='r', newline='') as Log:  # read datafile and store in dict self.data
            header = Log.readline()[:-1].split(',')
            self.data = {key: [] for key in header}
            for idx, row in enumerate(csv.DictReader(Log, fieldnames=header)):
                for key in self.data.keys():
                    self.data[key].append(row[key].strip())
        for key in ['xmax', 'xmin', 'ymax', 'ymin']:
            self.data[key] = [int(a) for a in self.data[key]]

    def get_labeled_images(self, label, create_non_label_folder=True):
        # name and create folders for the images to save
        folder_label_true = self.dataFolder+label
        self.make_dir(folder_label_true)
        if create_non_label_folder:
            folder_label_false = self.dataFolder+'non_'+label
            self.make_dir(folder_label_false)

        # iterate through images
        list_of_images = np.unique(self.data['Frame'])
        for img in list_of_images:
            bb_list = self.get_bboxes_of_image(img, label)
            self.save_images(img, bb_list, folder_label_true)



    def save_images(self, img_f, bboxes, folder):
        file = self.dataFolder+img_f
        img = cv2.imread(file)
        h,w = img.shape[:2]
        for idx, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
            # only save sub-images if sides are larger than 64px
            if (ymin-xmin)>=64 and (ymax-xmax)>=64 and ymin<h and ymax<w and xmin>=0 and xmax>=0:
                # Note: strange convention: pts for extraction must be (xmax, xmin), (ymax, ymin)
                slice = cv2.resize(img[xmin:ymin, xmax:ymax, :], self.img_size[:2])
                # cv2.rectangle(img, (xmax, xmin), (ymax, ymin), (255, 0, 0), 5)
                # plt.imshow(img)
                # plt.show()
                filename = folder+'/'+img_f[:-4]+'_'+str(idx)+'.png'
                cv2.imwrite(filename, slice)

                #Todo: for every saved image, find a random area that is not within the bboxes and save it to non_label folder

    def get_bboxes_of_image(self, img, label):
        """get all bounding boxes of the passed image as a list [((xmin,ymin),(xmax,ymax)), ...]"""
        bb_list=[]
        for idx, image in enumerate(self.data['Frame']):
            if (image == img) and (self.data['Label'][idx] == label):
                bb_list.append((self.data['xmax'][idx], self.data['ymax'][idx], self.data['xmin'][idx],
                                                                                   self.data['ymin'][idx]))
        return bb_list


    def make_dir(self, dirname):
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        else:
            if os.listdir(dirname) != []:
                print('Beware: Dir is not empty...')




def main():

    extractData().run()
    #extractData().read_labels()
# executes main() if script is executed directly as the main function and not loaded as module
if __name__ == '__main__':
    main()