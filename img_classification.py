import numpy as np
import cyvlfeat.sift as cfs
import cyvlfeat.kmeans as kmeans
from skimage import io, color, filters
from glob import glob
import pickle
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def chi2_dist(p, q):
    return 0.5 * np.sum([(x - y) ** 2 / (x + y + 1e-10) for (x, y) in zip(p, q)])


def load_images(path_to_img_folders):
    """
    Loads all the images and converts to grayscale.

    Arguments:
        path_to_img_folders {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    directories = sorted(glob(path_to_img_folders+'*'))
    # directory_names = [x.rsplit("/", 1)[1] for x in glob(path_to_data+'*')]
    paths_test = directories[::2]
    paths_train = directories[1::2]
    images_test = [color.rgb2gray(io.imread(path_to_img)) for sublist in [
        glob(y + '/*.jpg') for y in paths_test] for path_to_img in sublist]
    images_train = [color.rgb2gray(io.imread(path_to_img)) for sublist in [
        glob(y + '/*.jpg') for y in paths_train] for path_to_img in sublist]
    return images_test, images_train


def feat_detect_and_description(images_train, sift_func, dsift_blur=0, **kwargs):
    descriptors_of_train_images = []
    for img in images_train:
        if dsift_blur > 0:
            img = filters.gaussian(img, sigma=dsift_blur)
        _, descriptors = sift_func(img, **kwargs)
        descriptors_of_train_images.append(descriptors)
    descriptors = np.concatenate(descriptors_of_train_images)
    print('#descriptors_total:', descriptors.shape)
    return descriptors


def dictionary_comp(descriptors, k, distance_measure='l2'):
    return kmeans.kmeans(descriptors.astype(float), k, distance=distance_measure, initialization='PLUSPLUS')


def feat_quantization_and_hist_comp(all_images, centers, sift_func, **kwargs):
    k = centers.shape[0]
    histogram_space = []
    for img in all_images:
        # get img descriptors (#keypoints, 128) with sift
        _, descriptors = sift_func(img, **kwargs)

        # get distance of each dscriptor to each cluster
        distances = distance.cdist(centers, descriptors)

        # using argmin get label (which cluster) for each decriptor of the img
        labels = np.argmin(distances, axis=0)

        # make histogram: count occurrences using np.bincount how often each cluster is assigned
        occurrences = np.bincount(labels)
        while occurrences.shape[0] < k:
            occurrences = np.concatenate((occurrences, [0]))

        # normalize the histogram, divide by sum of occurrences
        histogram = occurrences / occurrences.sum()

        # append to features list/nparray/smt
        histogram_space.append(histogram)
    histogram_space = np.array(histogram_space)
    return histogram_space


def knn_classifier(histogram_space, k_neigbors, distance_measure='minkowski'):
    test_image_hists = histogram_space[:200, :]
    train_image_hists = histogram_space[200:, :]
    neigh = KNeighborsClassifier(
        n_neighbors=k_neigbors, metric=distance_measure)
    neigh.fit(train_image_hists, np.repeat(np.arange(4), 100))
    return [neigh.predict([test_img]) for test_img in test_image_hists], [neigh.kneighbors([test_img]) for test_img in test_image_hists]
    # return [neigh.predict([test_img]) for test_img in test_image_hists], []


def calculate_knn_classifier_accuary(predictions, results):
    acc_airplanes = len([x for x in predictions[0:50] if x == 0]) / 50
    acc_cars = len([x for x in predictions[50:100] if x == 1]) / 50
    acc_faces = len([x for x in predictions[100:150] if x == 2]) / 50
    acc_motorbikes = len([x for x in predictions[150:200] if x == 3]) / 50
    acc_overall = (acc_airplanes + acc_cars + acc_faces + acc_motorbikes) / 4
    return acc_airplanes, acc_cars, acc_faces, acc_motorbikes, acc_overall


def svm_classifier(histogram_space, kernel='rbf'):
    test_image_hists = histogram_space[:200, :]
    train_image_hists = histogram_space[200:, :]
    clf = SVC(kernel=kernel)
    clf.fit(train_image_hists, np.repeat(np.arange(4), 100))
    return [clf.predict([test_img]) for test_img in test_image_hists], []
