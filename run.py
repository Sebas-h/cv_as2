import pickle
import time
from glob import glob
import os

import cyvlfeat.sift as cfs
import matplotlib.pyplot as plt
import numpy as np

import img_classification as ic


def run(path_to_img_folders, sift_func, kmeans_num_clusters=50, kmeans_dist='l2',
        classifier='knn', knn_neighbors=1, knn_dist='minkowski', svm_kernel='rbf', **siftkwargs):
    classifier_info = 'kmeans_k='+str(kmeans_num_clusters)
    
    # LOAD IMAGES
    images_test, images_train = ic.load_images(path_to_img_folders)

    # FEATURES DETECTION AND DESCRIPTION
    training_img_descriptors = ic.feat_detect_and_description(
        images_train, sift_func, **siftkwargs)

    # DICTIONARY COMPUTATION
    centers = ic.dictionary_comp(training_img_descriptors, kmeans_num_clusters)

    # FEATURE QUANTIZATION AND HISTOGRAM COMPUTATION
    histogram_space = ic.feat_quantization_and_hist_comp(
        images_test + images_train, centers, sift_func, **siftkwargs)

    # CLASSIFIER TRAINING
    if classifier == 'svm':
        predictions, nearest_neighbors = ic.svm_classifier(histogram_space, kernel=svm_kernel)
        classifier_info  += '&svm_kernel='+svm_kernel
    else:
        predictions, nearest_neighbors = ic.knn_classifier(
            histogram_space, k_neigbors=knn_neighbors, distance_measure=knn_dist)
        classifier_info += '&knn_k='+str(knn_neighbors)+'&dist_measure='+knn_dist

    # CALCULATE ACCURACY
    accs = ic.calculate_knn_classifier_accuary(predictions, nearest_neighbors)
    print('=split("', str(sift_func).split(' ')[
          1], ',', classifier_info, ',', str(siftkwargs), '",",")')
    print('=split("', accs[0], ',', accs[1], ',',
          accs[2], ',', accs[3], ',', accs[4], '",",")')


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))[:-4] + '/data/data/'
    kmeans_k = 400
    knn_k = 5

    # SIFT
    # run(path, cfs.sift, kmeans_num_clusters=kmeans_k, knn_neighbors=knn_k,
    #     compute_descriptor=True, n_levels=12)

    # DSIFT
    run(path, cfs.dsift, kmeans_num_clusters=kmeans_k, knn_neighbors=knn_k,
        fast=True, step=10)

