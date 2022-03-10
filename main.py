import cv2
import numpy as np
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
    # FLANN - fast library for approximate nearest neighbours
    index_params = dict(algorithm=cv2.DESCRIPTOR_MATCHER_FLANNBASED, trees=5)
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


def warp_images(img1, img2, H, blending_func=None):
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

    # Just image 2
    filled_result = cv2.warpPerspective(img2, Ht.dot(H), (xmax - xmin, ymax - ymin), flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_REPLICATE)
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax - xmin, ymax - ymin), flags=cv2.INTER_NEAREST)

    if blending_func is None:
        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1
    else:
        img1_full = np.zeros_like(result, dtype=np.uint8)
        img1_full[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

        intersection = np.min(np.concatenate((img1_full, result), axis=2), axis=2)
        y, x = np.nonzero(intersection)
        x0, x1 = min(x), max(x)
        y0, y1 = min(y), max(y)

        # EXCLUSIVE
        filled_intersection_region = filled_result[y0: y1 + 1, x0: x1 + 1]

        # filling the gaps in the intersection regions with replicated pixels from 2nd image
        result[y0: y1 + 1, x0: x1 + 1] = filled_intersection_region

        result = blending_func(img1_full, result, (y0, y1 + 1, x0, x1 + 1))

    # both images
    return result


def _calc_center(img):
    if len(img.shape) > 2 or img.shape[2] != 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    y, x = np.nonzero(img)
    x0, x1 = min(x), max(x)
    y0, y1 = min(y), max(y)

    cx = np.mean((x0, x1))
    cy = np.mean((y0, y1))

    return cx, cy


def apply_feathering(img1_full, result_img, overlapping_region_coords):
    result_img = np.copy(result_img)
    # EXCLUSIVE
    y0, y1, x0, x1 = overlapping_region_coords

    cx1, cy1 = _calc_center(img1_full)
    cx2, cy2 = _calc_center(result_img)

    is_left = cx1 < cx2
    is_top = cy1 < cy2

    overlap_width = x1 - x0
    overlap_height = y1 - y0

    weights_horiz = np.repeat(np.linspace(0, 1, overlap_width).reshape(1, -1), overlap_height, axis=0)
    weights_vert = np.repeat(np.linspace(0, 1, overlap_height).reshape(-1, 1), overlap_width, axis=1)

    angle_cos = np.abs(np.cos(np.arctan2(cy2 - cy1, cx2 - cx1)))

    W = angle_cos * weights_horiz + (1 - angle_cos) * weights_vert

    if is_left:
        W = W[:, ::-1]
    if is_top:
        W = W[::-1, :]

    # repeat for 3 channels
    W = W[..., np.newaxis].repeat(3, axis=2)

    intersection_img1 = img1_full[y0: y1, x0: x1]
    intersection_result = result_img[y0: y1, x0: x1]

    intersection_img1 = np.where(intersection_img1 != 0, intersection_img1, intersection_result)
    intersection_result = np.where(intersection_result != 0, intersection_result, intersection_img1)

    # put img1 into result_img
    result_img = np.where(img1_full != 0, img1_full, result_img)

    # fill the intersection properly
    result_img[y0: y1, x0: x1] = (intersection_img1 * W + (1 - W) * intersection_result).astype(np.uint8)

    return result_img


def stitch_two_images(query_path, train_path, matching_mode, use_feathering=True):
    query_img = cv2.imread(query_path)
    train_img = cv2.imread(train_path)

    if matching_mode == 'baseline':
        points1, desc1 = extract_points(query_img)
        points2, desc2 = extract_points(train_img)

        points1, points2 = match_points(points1, points2, desc1, desc2)
    elif matching_mode == 'superglue':
        input_pairs = [[query_path, train_path]]
        points1, points2 = SuperGlueRun.match_pairs(input_pairs=input_pairs, output_dir='./output_keypoints',
                                                    superglue="outdoor", max_keypoints=2048, resize_float=True,
                                                    resize=[-1])
    else:
        raise ValueError(f"Unknown feature extraction method: {matching_mode}")

    projective_matrix, _mask = cv2.findHomography(points2, points1, cv2.RANSAC, 0.5, maxIters=1000)
    blending_func = apply_feathering if use_feathering else None

    return warp_images(query_img, train_img, projective_matrix, blending_func=blending_func)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image stitching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--matching_mode', type=str, default='baseline', choices=['baseline', 'superglue']
    )
    parser.add_argument(
        'img1', type=str, help='Path to image 1.',
    )
    parser.add_argument(
        'img2', type=str, help='Path to image 2.',
    )
    parser.add_argument(
        '-o', '--output', type=str, required=True, help='Path where the output panorama will be saved.',
        metavar='OUTPUT_IMAGE_PATH'
    )
    parser.add_argument(
        '--feathering', action=argparse.BooleanOptionalAction, default=True,
        help='Whether to apply feathering blending after stitching.',
    )
    args = parser.parse_args()

    query_path = args.img1  # 'image_pairs/image pairs_04_01.jpg'
    train_path = args.img2  # 'image_pairs/image pairs_04_02.jpg'

    result = stitch_two_images(query_path, train_path, args.matching_mode, args.feathering)

    cv2.imwrite(
        args.output,  # '04_feathering_special.png',
        result
    )
