import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import SuperGluePretrainedNetwork.SuperGlueRun as SuperGlueRun

MIN_MATCH_COUNT = 10


def extract_points(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kps = sift.detect(gray, mask=None)
    kps, descriptors = sift.compute(gray, kps)

    # (x, y)
    return kps, descriptors


def match_points(kps1, kps2, desc1, desc2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < MIN_MATCH_COUNT:
        print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
        exit(0)

    valid_points1 = [kps1[match.queryIdx].pt for match in good_matches]
    valid_points2 = [kps2[match.trainIdx].pt for match in good_matches]

    return np.array(valid_points1), np.array(valid_points2)


def warpTwoImages(img1, img2, H):
    # https://stackoverflow.com/a/20355545
    # just need to calculate the size properly so that both images fit on the screen
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    pts2_ = cv2.perspectiveTransform(pts2, H)

    pts = np.concatenate((pts1, pts2_), axis=0)

    [xmin, ymin] = np.int32(pts.min(axis=0).flatten() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).flatten() + 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax - xmin, ymax - ymin))
    result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

    return result


def stitch_two_images(queryPath, trainPath, matching_mode):
    queryImg = cv2.imread(queryPath)
    trainImg = cv2.imread(trainPath)

    if matching_mode == 'baseline':
        points1, desc1 = extract_points(queryImg)
        points2, desc2 = extract_points(trainImg)

        points1, points2 = match_points(points1, points2, desc1, desc2)
    elif matching_mode == 'superglue':
        input_pairs = [[queryPath, trainPath]]
        points1, points2 = SuperGlueRun.match_pairs(input_pairs=input_pairs, output_dir='./output_keypoints', superglue="outdoor", max_keypoints=2048, resize_float=True, resize=[-1])

    # TODO: try different algorithms here (supports RANSAC, LMEDS and RHO)

    # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
    projective_matrix, _mask = cv2.findHomography(points2, points1, cv2.RANSAC, 0.5, maxIters=1000)

    result = warpTwoImages(queryImg, trainImg, projective_matrix)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 10))
    plt.imshow(result)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image stitching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--matching_mode', type=str, default='baseline',
        help='baseline/superglue')
    opt = parser.parse_args()

    queryPath = 'image_pairs/image pairs_02_01.jpg'
    trainPath = 'image_pairs/image pairs_02_02.jpg'

    stitch_two_images(queryPath, trainPath, opt.matching_mode)
