from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch


from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)

def match_pairs(input_pairs = 'assets/scannet_sample_pairs_with_gt.txt',
output_dir = 'dump_match_pairs/', max_length = -1, resize = [640, 480], resize_float = False, superglue = 'indoor',
max_keypoints = 1024, keypoint_threshold = 0.005, nms_radius = 4, sinkhorn_iterations = 20, match_threshold = 0.2,
viz = False,  fast_viz = False, show_keypoints = False, viz_extension = 'png',
opencv_display = False, force_cpu = False):
    '''
    Image pair matching and pose evaluation with SuperGlue
    Input args:
    -input_pairs: Path to the list of image pairs
    -output_dir: Path to the directory in which the .npz results and optionally, the visualization images are written
    -max_length: Maximum number of pairs to evaluate
    -resize: Resize the input image before running inference. If two numbers, resize to the exact dimensions, if one number, resize the max dimension, if -1, do not resize
    -resize_float: Resize the image after casting uint8 to float
    -superglue: SuperGlue weights (indoor/outdoor)
    -max_keypoints: Maximum number of keypoints detected by Superpoint ('-1' keeps all keypoints)
    -keypoint_threshold: SuperPoint keypoint detector confidence threshold
    -nms_radius: SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)
    -sinkhorn_iterations: Number of Sinkhorn iterations performed by SuperGlue
    -match_threshold: SuperGlue match threshold
    -viz: Visualize the matches and dump the plots
    -fast_viz: Use faster image visualization with OpenCV instead of Matplotlib
    -show_keypoints: Plot the keypoints in addition to the matches
    -viz_extension: Visualization file extension. Use pdf for highest-quality (png/pdf)
    -opencv_display: Visualize via OpenCV before saving output images
    -force_cpu: Force pytorch to run in CPU mode
    '''

    assert not (opencv_display and not viz), 'Must use viz with opencv_display'
    assert not (opencv_display and not fast_viz), 'Cannot use opencv_display without fast_viz'
    assert not (fast_viz and not viz), 'Must use --viz with fast_viz'
    assert not (fast_viz and viz_extension == 'pdf'), 'Cannot use pdf extension with fast_viz'

    if len(resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            resize[0], resize[1]))
    elif len(resize) == 1 and resize[0] > 0:
        print('Will resize max dimension to {}'.format(resize[0]))
    elif len(resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')



    pairs = input_pairs

    if max_length > -1:
        pairs = pairs[0:np.min([len(pairs), max_length])]


    

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    output_dir = Path(output_dir)

    timer = AverageTimer(newline=True)
    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, viz_extension)

        # Handle --cache logic.
        do_match = True
        do_viz = viz

        if not (do_match or do_viz):
            timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
            continue

        # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        # Load the image pair.
        image0, inp0, scales0 = read_image(
            input_pairs[0][0], device, resize, rot0, resize_float)
        image1, inp1, scales1 = read_image(
            input_pairs[0][1], device, resize, rot1, resize_float)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_pairs[0][0], input_pairs[0][1]))
            exit(1)
        timer.update('load_image')

        if do_match:
            # Perform the matching.
            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            timer.update('matcher')


        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        if do_viz:
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, show_keypoints,
                fast_viz, opencv_display, 'Matches', small_text)
            timer.update('viz_match')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
        
        return mkpts0, mkpts1