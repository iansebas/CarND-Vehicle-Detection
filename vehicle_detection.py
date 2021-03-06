#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This program takes an image or video and finds the lane line on the road """

import argparse
import os
import time

import pickle
import glob

import numpy as np
import cv2

from moviepy.editor import VideoFileClip

from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import svm, grid_search
import multiprocessing

import _pickle as cPickle


######################
## HELPER FUNCTIONS ##
######################

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, (int(bbox[0,0]),int(bbox[0,1])), (int(bbox[1,0]),int(bbox[1,1])), color, thick)
    # Return the image copy with boxes drawn
    return draw_img
    
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features
    

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def find_cars(img, box_list, ystart, ystop, scale, svc, X_scaler, orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32,32), hist_bins=32, debug=False):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 3  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)) 
            test_prediction = svc.predict(test_features)   

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                if debug:
                    print("Vehicle Detcted!")
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                else:
                    points = np.array([[xbox_left, ytop_draw+ystart],[xbox_left+win_draw,ytop_draw+win_draw+ystart]])
                    box_list.append(points)
                
    if debug:
        return draw_img
    else:
        return box_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold=5):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def single_img_features(img, orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32):

    img = img.astype(np.float32)/255

    img = convert_color(img, conv='RGB2YCrCb')
    spatial_features = bin_spatial(img, size=spatial_size)
    hist_features = color_hist(img, nbins=hist_bins)

    ch1 = img[:,:,0]
    ch2 = img[:,:,1]
    ch3 = img[:,:,2]
    hog_feat1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=True)
    hog_feat2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=True)
    hog_feat3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=True)
    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

    features = np.hstack((spatial_features, hist_features, hog_features)).astype(np.float64).reshape(-1)

    return features


def load_data(X_raw,Y_raw,filepath,y_val):

    print("Starting to Load {}".format(filepath))

    images = glob.glob(filepath)
    for idx, fname in enumerate(images):
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        X_raw.append(image)
        Y_raw.append(y_val)

    print("Finished Loading {}".format(filepath))

    return X_raw, Y_raw

def get_raw_data():
    
    X_raw = []
    Y_raw = []

    ####################
    ## Vehicle Labels ##
    ####################

    X_raw, Y_raw = load_data(X_raw,Y_raw,"vehicles/GTI_Far/*.png",1)

    X_raw, Y_raw = load_data(X_raw,Y_raw,"vehicles/GTI_Left/*.png",1)

    X_raw, Y_raw = load_data(X_raw,Y_raw,"vehicles/GTI_MiddleClose/*.png",1)

    X_raw, Y_raw = load_data(X_raw,Y_raw,"vehicles/GTI_Right/*.png",1)

    X_raw, Y_raw = load_data(X_raw,Y_raw,"vehicles/KITTI_extracted/*.png",1)


    ####################
    ## Other Labels   ##
    ####################

    X_raw, Y_raw = load_data(X_raw,Y_raw,"non-vehicles/GTI/*.png",0)

    X_raw, Y_raw = load_data(X_raw,Y_raw,"non-vehicles/Extras/*.png",0)

    #######################
    ## Convert to Array  ##
    #######################

    X_raw = np.array(X_raw)
    Y_raw = np.array(Y_raw)

    return X_raw, Y_raw

def get_data():

    print("\n\nLoading Data")

    X_raw, Y_raw = get_raw_data()

    print("Getting Features from Raw Data")

    X_train = []
    for i in range(X_raw.shape[0]):
        xx = single_img_features(X_raw[i])
        X_train.append(xx)

    X_train = np.array(X_train)
    Y_train = Y_raw


    # Shuffle data
    print("Shuffling Data")
    X_train, Y_train = shuffle(X_train, Y_train)

    print("X_train shape:{}".format(X_train.shape))
    print("Y_train shape:{}".format(Y_train.shape))

    print("\n\nFinished Loading Data")

    return X_train, Y_train




######################
## Main Definitions ##
######################

class Vehicle_Detector():

    def __init__(self):
        self.svc = svm.SVC()
        self.scaler = StandardScaler()
        self.y_size = 0
        self.x_size = 0

        self.frame = 0
        self.processed_frames = 0
        self.skip_frame = 5
        self.box_list_memory = []

    def train(self, X_train, Y_train, grid_search_enabled=False):

        print("\n\nStarted Training")
        t_1=time.time()

        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)

        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.4, random_state=0)
        if grid_search_enabled:
            parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10],  'gamma':[0.1, 1, 10]}
            svr = svm.SVC()
            self.svc = grid_search.GridSearchCV(svr, parameters, n_jobs=multiprocessing.cpu_count())
        self.svc.fit(X_train, Y_train)

        # save the scaler
        with open('output_files/scaler.pkl', 'wb') as fid:
            cPickle.dump(self.scaler, fid) 
        # save the classifier
        with open('output_files/model.pkl', 'wb') as fid:
            cPickle.dump(self.svc, fid)    

        if grid_search_enabled:
            print("Best parameters set found on development set: {}".format(self.svc.best_params_))
        print("Cross Val Score : {}".format(self.svc.score(X_test, Y_test)))


        t_2=time.time()
        print("Finished Training")
        print("Training took {} seconds".format(round(t_2-t_1, 5)))

    def load_model(self):
        with open('output_files/scaler.pkl', 'rb') as fid:
            self.scaler = cPickle.load(fid)
        with open('output_files/model.pkl', 'rb') as fid:
            self.svc = cPickle.load(fid)

    def find_all_cars(self,img):

        y_size = img.shape[0]
        x_size = img.shape[1]

        box_list = []

        # Iterate over scales
        box_list = find_cars(img, box_list, y_size*0.5, y_size*0.7, 1, self.svc, self.scaler)
        box_list = find_cars(img, box_list, y_size*0.55, y_size*0.8, 2, self.svc, self.scaler)
        box_list = find_cars(img, box_list, y_size*0.65, y_size, 3, self.svc, self.scaler)
        box_list = find_cars(img, box_list, y_size*0.5, y_size, 4, self.svc, self.scaler)

        return box_list

    def process_image(self,img):

        if self.frame%self.skip_frame == 0 or self.frame < self.skip_frame :

            # Main function
            box_list = self.find_all_cars(img)

            self.box_list_memory = self.box_list_memory + box_list

            self.processed_frames += 1


        # Add heat to each box in box list
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        pos = int(len(self.box_list_memory)/self.processed_frames*self.skip_frame*3) if self.frame > self.skip_frame*3 else len(self.box_list_memory)-1
        heat = add_heat(heat,self.box_list_memory[-pos:])

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,2) if self.frame > self.skip_frame else apply_threshold(heat,1)

        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)



        self.frame += 1

        return draw_img

    def find_vehicles_image(self, filepath, save_result = True):

        # Reading in an image.
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.frame = 0
        self.box_list_memory = []

        result = self.process_image(image)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


        # Save Result in cv2 format (BGR)
        if save_result:
            pos = filepath.rfind('/')
            savepath = "output_files/"+ filepath[pos:]
            print("Saving image at: {}".format(savepath))
            cv2.imwrite(savepath,result)


    def find_vehicles_video(self, filepath, save_result = True):
        clip = VideoFileClip(filepath)
        
        self.frame = 0
        self.box_list_memory = []

        result_clip = clip.fl_image(self.process_image)        

        # Save Result 
        if save_result:
            pos = filepath.rfind('/')
            savepath = "output_files/"+filepath[pos:]
            print("\n Saving video at: {}".format(savepath))
            result_clip.write_videofile(savepath, audio=False)

    def debug(self,filepath):

        t_1=time.time()

        image = cv2.imread(filepath)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img2 = np.copy(img)
        y_size = img.shape[0]
        x_size = img.shape[1]
        cv2.line(img2, (0,int(y_size*0.5)), (int(x_size),int(y_size*0.5)), (0, 255, 255), 3)
        cv2.line(img2, (0,int(y_size*0.6)), (int(x_size),int(y_size*0.6)), (0, 255, 255), 3)
        cv2.line(img2, (0,int(y_size*0.7)), (int(x_size),int(y_size*0.7)), (0, 255, 255), 3)
        cv2.line(img2, (0,int(y_size*0.8)), (int(x_size),int(y_size*0.8)), (0, 255, 255), 3)
        cv2.line(img2, (0,int(y_size*0.9)), (int(x_size),int(y_size*0.9)), (0, 255, 255), 3)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        cv2.imwrite("figures/exploration.png",img2)

        box_list = self.find_all_cars(img)
        draw_img = draw_boxes(img, box_list)
        draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("figures/raw_predictions.png",draw_img)

        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = add_heat(heat,box_list)
        #heat = apply_threshold(heat,1)

        heatmap = np.clip(heat, 0, 255)
        cv2.imwrite("figures/heatmap.png",heatmap*255.0/heatmap.max())

        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)

        draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("figures/final_predictions.png",draw_img)

        t_2=time.time()

        print("\nProcessing took {} seconds".format(round(t_2-t_1, 5)))


    def test(self,already_trained=False):
        if not already_trained:
            X_train, Y_train = get_data()
            self.train(X_train, Y_train)
        else:
            self.load_model()
        print("\nSVC Param: {}".format(self.svc.get_params()))


        self.debug("test_files/test6.jpg")

        images = glob.glob("test_files/*.jpg")
        for idx, fname in enumerate(images):
            print("\nTesting on image {}".format(fname))
            t_1=time.time()
            self.find_vehicles_image(fname)
            t_2=time.time()
            print("Prediction took {} seconds".format(round(t_2-t_1, 5)))

        #self.find_vehicles_video("test_files/test_video.mp4")
        self.find_vehicles_video("test_files/project_video.mp4")



if __name__ == '__main__':

    detector = Vehicle_Detector()

    detector.test()
    
