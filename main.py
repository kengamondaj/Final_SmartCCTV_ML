from __future__ import division, print_function, absolute_import

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import sys
import cv2
import csv
import tensorflow as tf
import tensorflow.contrib.slim as slim

import pandas as pd
import time
from tqdm import tqdm

from collections import defaultdict
from io import StringIO

import argparse
from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from time import sleep


# sys.path.insert(0, os.path.abspath("./"))
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import math
from math import factorial
from deep_sort.iou_matching import iou
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
from numpy import asarray ,expand_dims
# from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist

from mtcnn.mtcnn import MTCNN

import requests
from pandas.io.json import json_normalize
import base64
# ================================================================================

def detection(sess2, video_path ,face_cap ):
    points_objs = []
    start = time.time()
    
    id_frame = 1
    
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    detector = MTCNN()

    pbar = tqdm(total=length)
    while(True):
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        image_np = np.array(frame)
        if(image_np.shape == ()): break

        print('Frame ID:', id_frame, '\tTime:', '{0:.2f}'.format(time.time()-start), 'seconds')

        image_np_expanded = np.expand_dims(image_np, axis=0)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess2.run([boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        count_boxes = 0
        thresh = 0.5
        max_boxes = 10
        
        for i, c in enumerate(classes):
            if (c == 1 and (scores[i] > thresh) and (count_boxes < max_boxes)):
                im_height = image_np.shape[0]
                im_width = image_np.shape[1]
                ymin, xmin, ymax, xmax = boxes[i]
                
                (left, right, top, bottom) = (int(xmin*im_width),  int(xmax*im_width),
                                              int(ymin*im_height), int(ymax*im_height))
                
                xF, yF, widthF, heightF = 0,0,0,0
                if face_cap :
                    # detect faces in the image
                    people = image_np[top:bottom,left:right]
                    people = people[...,::-1]
                    face = detector.detect_faces(people)
                    
                    if len(face) != 0 :
                        if face[0]['confidence'] >= 0.8:
                            xF, yF, widthF, heightF = face[0]['box']
                            xF = int(xF*0.9)
                            yF = int(yF*0.9)
                            widthF = int(widthF*1.25)
                            heightF = int(heightF*1.2)
                            if widthF > 100 and heightF > 130 :
                                img = people[yF:yF+heightF,xF:xF+widthF]
                                img = img[...,::-1]
                                nameF = 'face/'+str(id_frame)+'.jpg'
                                cv2.imwrite(nameF,img)
                            else : 
                                xF, yF, widthF, heightF = 0,0,0,0

                points_objs.append([
                    id_frame, -1,
                    left, top, right-left, bottom-top,
                    scores[i],
                    -1,-1,-1,
                    xF+left, yF+top, widthF, heightF,-1
                ])
                count_boxes += 1

        id_frame += 1
        pbar.update(1)
    pbar.close()

    cap.release()
    cv2.destroyAllWindows()
    if len(points_objs) == 0 :
        points_objs.append([
                    0, -1,
                    0, 0,0,0,
                    1,
                    -1,-1,-1,
                    0, 0, 0, 0,-1
                ])
    # write detection
    return np.array(points_objs)
   



def _batch_norm_fn(x, scope=None):
    if scope is None:
        scope = tf.get_variable_scope().name + "/bn"
    return slim.batch_norm(x, scope=scope)


def create_link(
        incoming, network_builder, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(stddev=1e-3),
        regularizer=None, is_first=False, summarize_activations=True):
    if is_first:
        network = incoming
    else:
        network = _batch_norm_fn(incoming, scope=scope + "/bn")
        network = nonlinearity(network)
        if summarize_activations:
            tf.summary.histogram(scope+"/activations", network)

    pre_block_network = network
    post_block_network = network_builder(pre_block_network, scope)

    incoming_dim = pre_block_network.get_shape().as_list()[-1]
    outgoing_dim = post_block_network.get_shape().as_list()[-1]
    if incoming_dim != outgoing_dim:
        assert outgoing_dim == 2 * incoming_dim, \
            "%d != %d" % (outgoing_dim, 2 * incoming)
        projection = slim.conv2d(
            incoming, outgoing_dim, 1, 2, padding="SAME", activation_fn=None,
            scope=scope+"/projection", weights_initializer=weights_initializer,
            biases_initializer=None, weights_regularizer=regularizer)
        network = projection + post_block_network
    else:
        network = incoming + post_block_network
    return network


def create_inner_block(
        incoming, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(1e-3),
        bias_initializer=tf.zeros_initializer(), regularizer=None,
        increase_dim=False, summarize_activations=True):
    n = incoming.get_shape().as_list()[-1]
    stride = 1
    if increase_dim:
        n *= 2
        stride = 2

    incoming = slim.conv2d(
        incoming, n, [3, 3], stride, activation_fn=nonlinearity, padding="SAME",
        normalizer_fn=_batch_norm_fn, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/1")
    if summarize_activations:
        tf.summary.histogram(incoming.name + "/activations", incoming)

    incoming = slim.dropout(incoming, keep_prob=0.6)

    incoming = slim.conv2d(
        incoming, n, [3, 3], 1, activation_fn=None, padding="SAME",
        normalizer_fn=None, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/2")
    return incoming


def residual_block(incoming, scope, nonlinearity=tf.nn.elu,
                   weights_initializer=tf.truncated_normal_initializer(1e3),
                   bias_initializer=tf.zeros_initializer(), regularizer=None,
                   increase_dim=False, is_first=False,
                   summarize_activations=True):

    def network_builder(x, s):
        return create_inner_block(
            x, s, nonlinearity, weights_initializer, bias_initializer,
            regularizer, increase_dim, summarize_activations)

    return create_link(
        incoming, network_builder, scope, nonlinearity, weights_initializer,
        regularizer, is_first, summarize_activations)


def _create_network(incoming, num_classes, reuse=tf.AUTO_REUSE, l2_normalize=True,
                   create_summaries=True, weight_decay=1e-8):
    nonlinearity = tf.nn.elu
    conv_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    conv_bias_init = tf.zeros_initializer()
    conv_regularizer = slim.l2_regularizer(weight_decay)
    fc_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    fc_bias_init = tf.zeros_initializer()
    fc_regularizer = slim.l2_regularizer(weight_decay)

    def batch_norm_fn(x):
        return slim.batch_norm(x, scope=tf.get_variable_scope().name + "/bn")

    network = incoming
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_1",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)
        tf.summary.image("conv1_1/weights", tf.transpose(
            slim.get_variables("conv1_1/weights:0")[0], [3, 0, 1, 2]),
                         max_images=128)
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_2",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    if create_summaries:
        tf.summary.histogram(network.name + "/activations", network)

    network = slim.max_pool2d(network, [3, 3], [2, 2], scope="pool1")

    network = residual_block(
        network, "conv2_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False, is_first=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv2_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_block(
        network, "conv3_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv3_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    network = residual_block(
        network, "conv4_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True,
        summarize_activations=create_summaries)
    network = residual_block(
        network, "conv4_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False,
        summarize_activations=create_summaries)

    feature_dim = network.get_shape().as_list()[-1]
    print("feature dimensionality: ", feature_dim)
    network = slim.flatten(network)

    network = slim.dropout(network, keep_prob=0.6)
    network = slim.fully_connected(
        network, feature_dim, activation_fn=nonlinearity,
        normalizer_fn=batch_norm_fn, weights_regularizer=fc_regularizer,
        scope="fc1", weights_initializer=fc_weight_init,
        biases_initializer=fc_bias_init)

    features = network

    if l2_normalize:
        # Features in rows, normalize axis 1.
        features = slim.batch_norm(features, scope="ball", reuse=reuse)
        feature_norm = tf.sqrt(
            tf.constant(1e-8, tf.float32) +
            tf.reduce_sum(tf.square(features), [1], keep_dims=True))
        features = features / feature_norm

        with slim.variable_scope.variable_scope("ball", reuse=reuse):
            weights = slim.model_variable(
                "mean_vectors", (feature_dim, num_classes),
                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer=None)
            scale = slim.model_variable(
                "scale", (num_classes, ), tf.float32,
                tf.constant_initializer(0., tf.float32), regularizer=None)
            if create_summaries:
                tf.summary.histogram("scale", scale)

            scale = tf.nn.softplus(scale)

        # Each mean vector in columns, normalize axis 0.
        weight_norm = tf.sqrt(
            tf.constant(1e-8, tf.float32) +
            tf.reduce_sum(tf.square(weights), [0], keep_dims=True))
        logits = scale * tf.matmul(features, weights / weight_norm)

    else:
        logits = slim.fully_connected(
            features, num_classes, activation_fn=None,
            normalizer_fn=None, weights_regularizer=fc_regularizer,
            scope="softmax", weights_initializer=fc_weight_init,
            biases_initializer=fc_bias_init)

    return features, logits


def _network_factory(num_classes, is_training, weight_decay=1e-8):

    def factory_fn(image, reuse, l2_normalize):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
                with slim.arg_scope([slim.conv2d, slim.fully_connected,
                                     slim.batch_norm, slim.layer_norm],
                                    reuse=reuse):
                    features, logits = _create_network(
                        image, num_classes, l2_normalize=l2_normalize,
                        reuse=reuse, create_summaries=is_training,
                        weight_decay=weight_decay)
                    return features, logits

    return factory_fn


def _preprocess(image, is_training=False, enable_more_augmentation=True):
    image = image[:, :, ::-1]  # BGR to RGB
    if is_training:
        image = tf.image.random_flip_left_right(image)
        if enable_more_augmentation:
            image = tf.image.random_brightness(image, max_delta=50)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, patch_shape[::-1])

    return image


def _create_image_encoder(preprocess_fn, factory_fn, image_shape, batch_size=32,
                         session=None, checkpoint_path=None,
                         loss_mode="cosine"):
    image_var = tf.placeholder(tf.uint8, (None, ) + image_shape)

    preprocessed_image_var = tf.map_fn(
        lambda x: preprocess_fn(x, is_training=False),
        tf.cast(image_var, tf.float32))

    l2_normalize = loss_mode == "cosine"
    feature_var, _ = factory_fn(
        preprocessed_image_var, l2_normalize=l2_normalize, reuse=tf.AUTO_REUSE)
    feature_dim = feature_var.get_shape().as_list()[-1]

    if session is None:
        session = tf.Session()
    if checkpoint_path is not None:
        slim.get_or_create_global_step()
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
            checkpoint_path, slim.get_variables_to_restore())
        session.run(init_assign_op, feed_dict=init_feed_dict)

    def encoder(data_x):
        out = np.zeros((len(data_x), feature_dim), np.float32)
        _run_in_batches(
            lambda x: session.run(feature_var, feed_dict=x),
            {image_var: data_x}, out, batch_size)
        return out

    return encoder


def create_image_encoder(model_filename, batch_size=32, loss_mode="cosine",
                         session=None):
    image_shape = 128, 64, 3
    factory_fn = _network_factory(num_classes=1501, is_training=False, weight_decay=1e-8)

    return _create_image_encoder(_preprocess, factory_fn, image_shape, batch_size, session,
        model_filename, loss_mode)


def create_box_encoder(model_filename, batch_size=32, loss_mode="cosine"):
    image_shape = 128, 64, 3
    image_encoder = create_image_encoder(model_filename, batch_size, loss_mode)

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches)

    return encoder


def generate_detections(encoder, video_dir, det_data ,test_video):    
    videos = os.listdir(video_dir)
    videos.sort()
    for video_name in videos:
        if(video_name != test_video and test_video != '' ): 
            continue

        print("Processing %s" % video_name)

        detections_in = det_data
        detections_out = []

        cap = cv2.VideoCapture(os.path.join(video_dir, video_name))

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in tqdm(range(min_frame_idx, max_frame_idx + 1)):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]
            ret, bgr_image = cap.read()
            features = encoder(bgr_image, rows[:, 2:6].copy())
            # edit
            rows = rows[:,0:14]
            detections_out += [np.r_[(row, feature)] for row, feature in zip(rows, features)]

        # feature_filename = os.path.join(feat_dir, "%s.npy" % video_name[:-4])
        # np.save(feature_filename, np.asarray(detections_out), allow_pickle=False)
        return np.asarray(detections_out)

def gather_sequence_info(video_name, video_path, feat_data):
    detections = feat_data

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    image_size = np.array(frame).shape[:-1] 
    # print(image_size)

    min_frame_idx = int(detections[:, 0].min())
    max_frame_idx = int(detections[:, 0].max())

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": video_name,
        "detections": detections,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": None
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature , f_bbox = row[2:6], row[6], row[14:] ,row[10:14]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature ,f_bbox))
    return detection_list


def join(df_track):
    prev_frame_idx = min(df_track['track_id'].index)
    results = []
    for frame_idx, currrent_row in df_track.iterrows():
        gap = frame_idx - prev_frame_idx
        if(gap > 1):
            results.append(str(prev_frame_idx)+' -> '+ str(frame_idx))
            currrent_row = np.array(currrent_row)
            previous_row = np.array(df_track.loc[prev_frame_idx].values)
            steps = (currrent_row - previous_row) / gap

            for i, frame in enumerate(range(prev_frame_idx+1,frame_idx)):
                df_track.loc[frame] = np.array(previous_row + (i+1) * steps).astype(int)

        prev_frame_idx = frame_idx
    df_track = df_track.sort_index()

    misses = np.squeeze(list(set(range(min(df_track.index), 
                                       max(df_track.index) + 1)).difference(df_track.index)))
    if(len(misses)==0 and len(results) > 0):
        print('Track:', int(df_track['track_id'].iloc[0]),', concatenation complete, ',results)
    elif(len(misses)!=0):
        print('Warning!! Frame:', int(df_track['track_id'].iloc[0]), ', concatenation incomplete\n')
    return df_track

def run_concatenate(results):

    df_all = pd.DataFrame(results)
    df = df_all.iloc[:,:15]
    df.columns = ['frame_id','track_id','xmin','ymin','width','height', 
                  'confidence','neg_1', 'neg_2', 'neg_3','xF','yF','widthF', 'heightF', 'neg_5']
    df.index = df['frame_id']
    df = df.drop(['frame_id'], axis=1)

    concat = []
    From, To = min(df['track_id']), max(df['track_id'])+1
    for track_id in range(From, To):
        concat.append(join(df.loc[df['track_id']==track_id].copy()))
        
    df_concat = pd.concat(concat)
    df_concat = df_concat.sort_index()
    df_concat['filename'] = df_all.iloc[0,15]
    # df_concat.to_csv(concat_track_file, header=None)
    print('=================')
    return df_concat

def gather_sequence_info_2(video_name, video_path, feat_path):
    detections = np.load(feat_path)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    image_size = np.array(frame).shape[:-1] 
    # print(image_size)

    min_frame_idx = int(detections[:, 0].min())
    max_frame_idx = int(detections[:, 0].max())

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": video_name,
        "detections": detections,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": None
    }
    return seq_info

# ========================================================================================================


def capture(video_path, cap_dir, results, seq_info, is_plot=False):

    if os.path.exists(cap_dir):
        shutil.rmtree(cap_dir)
    os.makedirs(cap_dir)

    cap = cv2.VideoCapture(video_path)

    N_track = int(max(results[:,1]))
    subplot_x = 6
    subplot_y = int(math.ceil(N_track/subplot_x))
    print('Total Tracks:', N_track)
    print('Subplot', subplot_y, subplot_x)

    image_size = seq_info['image_size']
    points = {}
    captured = []

    with tf.Session() as sess:
        for frame_idx in tqdm(range(
                            seq_info['min_frame_idx'], 
                            seq_info['max_frame_idx'] + 1), 'capturing output'):
        
            image_np = np.array(cap.read()[1])

            mask = results[:, 0].astype(np.int) == frame_idx
            track_ids = results[mask, 1].astype(np.int)
            boxes = results[mask, 2:6]

            for track_id, box in zip(track_ids, boxes):
                if(track_id not in captured):
                    captured.append(track_id)

                    l,t,w,h = np.array(box).astype(int)
                    if(l<0): l=0 # if xmin is negative 
                    if(t<0): t=0 # if ymin is negative

                    if(l+w > image_size[1]): w=image_size[1]-l # if xmax exceeds width
                    if(t+h > image_size[0]): h=image_size[0]-t # if ymax exceeds height

                    cropped_image = sess.run(tf.image.crop_to_bounding_box(image_np, t, l, h, w))
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

                    img = Image.fromarray(cropped_image)
                    img.save(os.path.join(cap_dir, str(track_id)+'.jpg'))

                    if(is_plot):
                        plt.subplot(subplot_y, subplot_x, len(captured))
                        plt.imshow(cropped_image)
                        plt.title(str(track_id)+', '+str(frame_idx))

    cap.release()

    if(is_plot):
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.5, wspace=0.8)
        plt.show()

# ========================================================================================================
   
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2: # order should be less than or equal window-2
        raise TypeError("window_size is too small for the polynomials order")
        
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    
    # pad the signal at the extremes with
    # values taken from the signal itself
    
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    
    return np.convolve( m[::-1], y, mode='valid')

# ========================================================================================================

def golay_filter(df_track, window_size=45, order=5):
    if(len(df_track) <= window_size):
        return df_track
    df_track[2] = savitzky_golay(df_track[2].values, window_size=window_size, order=order, deriv=0, rate=1)
    df_track[3] = savitzky_golay(df_track[3].values, window_size=window_size, order=order, deriv=0, rate=1)
    df_track[4] = savitzky_golay(df_track[4].values, window_size=window_size, order=order, deriv=0, rate=1)
    df_track[5] = savitzky_golay(df_track[5].values, window_size=window_size, order=order, deriv=0, rate=1)
    return df_track
    
def poly_interpolate(df_track):
    model = make_pipeline(PolynomialFeatures(5), Ridge(solver='svd'))
    X = np.array(df_track.index).reshape(-1, 1)
    df_track[2] = model.fit(X, df_track[2]).predict(X)
    df_track[3] = model.fit(X, df_track[3]).predict(X)
    df_track[4] = model.fit(X, df_track[4]).predict(X)
    df_track[5] = model.fit(X, df_track[5]).predict(X)
    return df_track

def moving_avg(df_track, window=5):
    df_haed = df_track[[2,3,4,5]][:window-1]
    df_tail = df_track[[2,3,4,5]].rolling(window=window).mean()[window-1:]
    df_track[[2,3,4,5]] = pd.concat([df_haed, df_tail], axis=0)
    return df_track

def smooth(df, smooth_method):
    polynomials = []
    From, To = min(df[1]), max(df[1])+1

    for track_id in range(From, To):
        df_track = df.loc[df[1]==track_id].copy()

        if(smooth_method == 'poly'): df_track = poly_interpolate(df_track)
        elif(smooth_method == 'moving'): df_track = moving_avg(df_track)
        elif(smooth_method == 'golay'): df_track = golay_filter(df_track)
            
        polynomials.append(df_track)

    df_smooth = pd.concat(polynomials)
    df_smooth = df_smooth.sort_index()
    return df_smooth
    # return df_smooth.values
    
    
############################################################################################



loss_mode = "cosine"

model = "resources/networks/mars-small128.ckpt-68577"

max_cosine_distances = [0.35,0.35,0.3,0.35]

nn_budget = None



# ==============================================================================
fs = []
count_human_lasts = []
metrics = []
trackers = []

for i in range(4):
    fs.append(create_box_encoder(model, batch_size=32, loss_mode=loss_mode))
    count_human_lasts.append([])
# --------------
    metrics.append(nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distances[i], nn_budget))
    trackers.append(Tracker(metrics[i], max_age=50, n_init=5))

# ==============================================================================

print ('loading model..')
PATH_TO_CKPT = 'object_detection/faster_rcnn_resnet101_coco_11_06_2017/frozen_inference_graph.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(os.path.join('object_detection','data', 'mscoco_label_map.pbtxt'))
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ==============================================================================

def run_process(sess2,video_name,f,count_human_last,metric,tracker,output,output_feat,cam,face_cap):

    all_results = pd.DataFrame([])

    video_dir = "dataset/videos"

    det_dir = "dataset/detections"

    feat_dir = "dataset/features"

    loss_mode = "cosine"

    model = "resources/networks/mars-small128.ckpt-68577"

    tracks_dir= "dataset/tracks"

    min_confidence = 0.8

    min_detection_height = 0

    nms_max_overlap = 1.0

    
    videos = os.listdir(video_dir)
    videos.sort()
    results = []
    feature_bbox = []

    if video_name in videos:
        video_path = os.path.join(video_dir, video_name)
        #dont need
        feat_path  = os.path.join(feat_dir, video_name[:-3]+'npy')
        print('Processing Video:', video_name + '..')
        det_data = detection(sess2, video_path=os.path.join(video_dir, video_name),face_cap = face_cap)
        det_data_temp = pd.DataFrame(det_data)
        det_data_temp.to_csv(det_dir+'/'+video_name[:-3]+'csv',index = False)
        # ==============================================================================
        feat_data = generate_detections(f, video_dir, det_data ,video_name)
        
        feature_filename = os.path.join(feat_dir, "%s.npy" % video_name[:-4])
        np.save(feature_filename, feat_data, allow_pickle=False)
        # ==============================================================================
        if len(feat_data) == 1 or det_data_temp[6].max() < min_confidence:
            feature_bbox.append(list(np.zeros(128)))
            results.append([0, 0, 0, 0, 0, 0,1,-1,-1,-1
                                    ,0, 0, 0, 0,-1,video_path])

        else :
            seq_info = gather_sequence_info(video_name, video_path, feat_data)
        
            print('Video Path:', video_path,'\tFeatures:', feat_path)

            def frame_callback(vis, frame_idx):
                # print("Processing frame %05d" % frame_idx)

                # Load image and generate detections.
                detections = create_detections(seq_info["detections"], frame_idx, min_detection_height)
                detections = [d for d in detections if d.confidence >= min_confidence]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

                # Update tracker.
                tracker.predict()
                tracker.update(detections)

                # Update visualization.
                count_human = vis.draw_trackers(tracker.tracks)

                # Store results.
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlwh()
                    f_bbox = track.f_bbox
                    bbox_feat = track.bbox_feat
                    ID = count_human.index(track.track_id) + 1
    #                 ID = str(cam)+'_'+str(ID)
                    results.append([frame_idx, ID, bbox[0], bbox[1], bbox[2], bbox[3],1,-1,-1,-1
                                    ,f_bbox[0], f_bbox[1], f_bbox[2], f_bbox[3],-1,video_path])
                    feature_bbox.append(bbox_feat)

            visualizer = visualization.NoVisualization( seq_info, count_human_last)
            visualizer.run(frame_callback)

    # sent (results) to database here6
    # results = run_concatenate(results)
    if len(results) < 1 :
        feature_bbox = []
        results = []
        feature_bbox.append(list(np.zeros(128)))
        results.append([0, 0, 0, 0, 0, 0,1,-1,-1,-1
                                ,0, 0, 0, 0,-1,video_path])

    results = pd.DataFrame(results)
    results.columns = range(results.shape[1])
    if results.shape[0] > 1 :
        results = smooth(results, smooth_method='golay')
    # edit
    print(results.shape)
    results[1] = str(cam)+'_' + results[1].astype(str)
    
    output_feat.put((int(cam), feature_bbox))
    output.put((cam, results))
    # sum for test
    # all_results = pd.concat([all_results, results])

    file_name = 'dataset/tracks/'+ video_name[:-3]+'csv'
    results.to_csv(file_name, header=None,index=False)

def sortFirst(val): 
    return val[0] 

def extract_face(filename, detector, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


def getEmbedding(resized):
    reshaped = resized.reshape(-1,160,160,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    # print(feed_dict)
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding

def get_face_embedding(video_dir,test_video,face_list):
    videos = os.listdir(video_dir)
    videos.sort()

    for video_name in videos:
        if(video_name != test_video and test_video != '' ): 
            continue

        print("Processing %s" % video_name)

        cap = cv2.VideoCapture(os.path.join(video_dir, video_name))
        face_embedding_out = []
        face_pics = []
        min_frame_idx = 0
        max_frame_idx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            temp_face_list = face_list[face_list[0] == frame_idx]
            ret, bgr_image = cap.read()
            if len(temp_face_list) != 0 :
                for index, row in temp_face_list.iterrows():
                    features = []
                    face_ids = []
                    bgr_image = np.asarray(bgr_image)
                    bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                    check_x_min = int(row[10])
                    if check_x_min < 0 :
                        check_x_min = 1
                    face_pixels = bgr_image[int(row[11]):int(row[11]+row[13]), check_x_min:int(row[10]+row[12])]
                    face_pics.append([row[1],face_pixels[...,::-1]])
                    face_pixels = Image.fromarray(face_pixels)
                    face_pixels = face_pixels.resize((160,160))
                    face_pixels = asarray(face_pixels)
                    prewhitened = facenet.prewhiten(face_pixels)
                    features.append(getEmbedding(prewhitened)[0])
                    face_ids.append(row[1])
            # edit
                    face_embedding_out += [np.r_[(face_ids, feature)] for face_id, feature in zip(face_ids, features)]

        
        return pd.DataFrame(face_embedding_out) , pd.DataFrame(face_pics)

def assign_RealPosition(df_track,Box_set,num_cam,cam_path,Real_position) :

    for index1, row1 in tqdm(df_track.iterrows()):
        min_dis = [10000000,0,'']
        df_temp = Box_set[Box_set['cam_id']==num_cam] 
        for index2, row2 in df_temp[df_temp['pixelColumn'].between((row1[2]+(row1[4]/2))-500, (row1[2]+(row1[4]/2))+500) & df_temp['pixelRow'].between((row1[3]+row1[5])-500, (row1[3]+row1[5])+500)].iterrows() :
            dis = pdist(np.array([[row2['pixelColumn'],row2['pixelRow']],[(row1[2]+(row1[4]/2)),(row1[3]+row1[5])]]))
            if dis < min_dis[0]:
                min_dis[0] = dis
                min_dis[1] = row2['block_id']
                min_dis[2] = row2['area_name']
        date = row1[15][-18:-14]  + '-'+ row1[15][-14:-12] + '-'+ row1[15][-12:-10]
        times = row1[15][-10:-8] + ':'+ row1[15][-8:-6] +':'+ row1[15][-6:-4]
        Real_position = pd.concat([Real_position,pd.DataFrame(data= {'user_id': [row1[1]], 'block_id': [min_dis[1]]
                                                                ,'area_name': [min_dis[2]], 'cam_id': [num_cam], 'video_name': cam_path+str(row1[15][-18:-4])
                                                                , 'date': [date], 'time': [times],'x': [row1[2]],'y': [row1[3]],'width': [row1[4]],'height': [row1[5]]})],sort=False)

    return Real_position 

def getFace(img,detector):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    face = detector.detect_faces(img)
    if len(face) != 0 :
        if face[0]['confidence'] >= 0.8:
            xF, yF, widthF, heightF = face[0]['box']
            xF = int(xF*0.9)
            yF = int(yF*0.9)
            widthF = int(widthF*1.25)
            heightF = int(heightF*1.2)
            
            cropped = img[yF:yF+heightF,xF:xF+widthF]
            resized = cv2.resize(cropped, (160,160),interpolation=cv2.INTER_CUBIC)
            prewhitened = facenet.prewhiten(resized)
            faces.append({'face':resized,'embedding':getEmbedding(prewhitened)})
            
    return faces

def all_api(new_face,api_path):
    facenet.load_model("20170512-110547/20170512-110547.pb")
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    
    face_embedding_ = []
    face_database = os.listdir('Member_database')
    for filename in face_database :
        face_ids = []
        temp_face_embeddings = []
        img = cv2.imread('Member_database/'+filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_pixels = getFace(img,detector)
        temp_face_embeddings.append(face_pixels[0]['embedding'][0])
        face_ids.append(filename[:-6])
        face_embedding_ += [np.r_[(face_ids, temp_face_embedding)] for face_id, temp_face_embedding in zip(face_ids, temp_face_embeddings)]

    face_embedding_ = pd.DataFrame(face_embedding_)
    face_embedding_ = pd.concat([face_embedding_[0],face_embedding_.drop(columns=[0]).astype(float)],axis=1)
    
    while 1 :
        # files = os.listdir('addface')
        newMember = requests.get(api_path+'/members/newMember')
        if len(newMember.text) > 10:
            newMember = newMember.json()
            user = requests.get(api_path+'/members/face/'+newMember['user_id']).json() 
            for num,pic in enumerate(user[0]['picture']):
                new_face_embedding_ = []
                if num < 3 :
                    face_ids = []
                    temp_face_embeddings = []
                    base64_img_bytes = pic.encode('utf-8')
                    filename = 'newMember/' + str(user[0]['user_id'])+'_'+str(num)+'.png'
                    with open(filename, 'wb') as file_to_save:
                        decoded_image_data = base64.decodebytes(base64_img_bytes)
                        file_to_save.write(decoded_image_data)

                    img = cv2.imread(filename)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face_pixels = getFace(img,detector)
                    temp_face_embeddings.append(face_pixels[0]['embedding'][0])
                    face_ids.append(user[0]['user_id'])
                    new_face_embedding_ += [np.r_[(face_ids, temp_face_embedding)] for face_id, temp_face_embedding in zip(face_ids, temp_face_embeddings)]
                    new_face.put(pd.DataFrame(new_face_embedding_))

                    new_face_embedding_ = pd.DataFrame(new_face_embedding_)
                    new_face_embedding_ = pd.concat([new_face_embedding_[0],new_face_embedding_.drop(columns=[0]).astype(float)],axis=1)
                    face_embedding_ = pd.concat([face_embedding_ , new_face_embedding_])


        image_ver = requests.get(api_path +'/search/faceVerify')
        if len(image_ver.text) > 100 :
            
            temp_face_embeddings = []
            search_face_embedding_ = []
            image_ver = image_ver.json()
            base64_img_bytes = image_ver[0]['picture'].encode('utf-8')

            # base64_img_bytes = str(image_ver.content)[2:].encode('utf-8')
            
            filename = 'searchface/'+str(int(time.time())) +'.png'
            with open(filename, 'wb') as file_to_save:
                decoded_image_data = base64.decodebytes(base64_img_bytes)
                file_to_save.write(decoded_image_data)

            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_pixels = getFace(img,detector)
            temp_face_embeddings.append(face_pixels[0]['embedding'][0])

            search_face_embedding_ += [np.r_[(temp_face_embedding)] for temp_face_embedding in  temp_face_embeddings]
            search_face_embedding_ = pd.DataFrame(search_face_embedding_)
            search_face_embedding_ = pd.concat([search_face_embedding_[0],search_face_embedding_.drop(columns=[0]).astype(float)],axis=1)
            search_face_embedding_.to_csv('kuyeayyyy.csv')

            resultt = pd.DataFrame(data= {'user_id': [], 'dis': []})

            if len(search_face_embedding_) != 0:
                for index0, row0 in search_face_embedding_.iterrows():
                    for index, row in face_embedding_.iterrows():
                        d = distance.euclidean(list(row[1:]), list(row0))
                        real_id = str(int(row[0]))
                        resultt = pd.concat([resultt,pd.DataFrame(data= {'user_id': [real_id], 'dis': [d]})])
                    resultt = resultt.groupby(['user_id'])['dis'].min().reset_index()
                    resultt = resultt.sort_values(by=['dis']).reset_index(drop=True)
                    
                resultt.to_csv('ppppppppppppppp.csv')
                if resultt['dis'][0] < 0.9:
                    # print(result.iloc[:3])
                    res = { 'tier1' : resultt.user_id[0],
                            'tier2' : resultt.user_id[1],
                            'tier3' : resultt.user_id[2],}
                    pd.DataFrame([resultt.user_id[0],resultt.user_id[1],resultt.user_id[2]]).to_csv('ggggggggg.csv')
                    url = api_path+'/search/resultVerify'
                    x = requests.post(url, json = res)
                    
                else :
                    res = {'tier1' : '',
                            'tier2' : '',
                            'tier3' : '',}
                    
                    url = api_path+'/search/resultVerify'
                    x = requests.post(url, json = res)
                    # print('unknown')


import requests
from pandas.io.json import json_normalize
from scipy.spatial import distance
import threading 
import multiprocessing as mp

output = mp.Queue()
output_feat = mp.Queue()
new_face = mp.Queue()

reid_dict = {}
reid_dict_2 = {}
reid_face_dict = {}
reid_life = 0

video_dir = "dataset/videos"

det_dir = "dataset/detections"

feat_dir = "dataset/features"

# data = requests.get('http://a393f731.ngrok.io/blocks/all').json()
# df = json_normalize(data)
# Real_position = pd.DataFrame(data= {'user_id': [], 'block_id': [],'area_name': [], 'cam_id': [], 'video_name': [], 'date': [], 'time': []})

api_path = 'http://cc9677bd.ngrok.io'

box_data = requests.get(api_path+'/blocks/all').json()
box_data = json_normalize(box_data)
box_data.to_csv('Box_setting.csv',index=False)
Box_set = pd.read_csv('Box_setting.csv')

face_filenames = os.listdir('face_database')
# facenet = load_model('model/facenet_keras.h5')
detector = MTCNN()

import facenet

sess = tf.Session()

# read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
facenet.load_model("20170512-110547/20170512-110547.pb")

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

face_embedding = []
unknown_id = 1


members_database = requests.get(api_path+'/members/get').json()

for mem in tqdm(members_database):
    user = requests.get(api_path+'/members/face/'+str(mem['user_id'])).json() 
    for num,pic in enumerate(user[0]['picture']):
        if num < 3 :
            face_ids = []
            temp_face_embeddings = []
            base64_img_bytes = pic.encode('utf-8')
            filename = 'Member_database/'+str(user[0]['user_id'])+'_'+str(num)+'.png'
            with open(filename, 'wb') as file_to_save:
                decoded_image_data = base64.decodebytes(base64_img_bytes)
                file_to_save.write(decoded_image_data)

            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_pixels = getFace(img,detector)
            temp_face_embeddings.append(face_pixels[0]['embedding'][0])
            face_ids.append(user[0]['user_id'])
            face_embedding += [np.r_[(face_ids, temp_face_embedding)] for face_id, temp_face_embedding in zip(face_ids, temp_face_embeddings)]

face_embedding = pd.DataFrame(face_embedding)

# for filename in face_filenames :
#     face_ids = []
#     temp_face_embeddings = []
#     img = cv2.imread('face_database/'+filename)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     face_pixels = getFace(img,detector)
#     temp_face_embeddings.append(face_pixels[0]['embedding'][0])
#     face_ids.append(user[0]['user_id'])
#     face_embedding += [np.r_[(face_ids, temp_face_embedding)] for face_id, temp_face_embedding in zip(face_ids, temp_face_embeddings)]

# face_embedding = pd.DataFrame(face_embedding)
# face_embedding.to_csv('fucker.csv')
face_embedding.to_csv('face.csv')
face_inden = threading.Thread(target=all_api, args=(new_face,api_path,)) 
face_inden.start() 


l1 = [] 
l2 = []
l3 = []
l4 = []

videos_list = os.listdir(video_dir)
videos_list.sort()

for name in videos_list :
    if name.startswith("121") :
        l1.append(name)
    elif name.startswith("120") :
        l2.append(name)
    elif name.startswith("220") :
        l3.append(name)
    elif name.startswith("122") :
        l4.append(name)

with detection_graph.as_default():
    sess2 = tf.Session(graph=detection_graph) 

for l1_v,l2_v,l3_v,l4_v  in zip(l1[1:],l2[1:],l3[1:],l4[1:]):
    print((l1_v,l2_v,l3_v,l4_v))
    t1 = threading.Thread(target=run_process, args=(sess2,l1_v,fs[0],count_human_lasts[0],metrics[0],trackers[0],output,output_feat,1,False,)) 
    t2 = threading.Thread(target=run_process, args=(sess2,l2_v,fs[1],count_human_lasts[1],metrics[1],trackers[1],output,output_feat,2,False,)) 
    t3 = threading.Thread(target=run_process, args=(sess2,l3_v,fs[2],count_human_lasts[2],metrics[2],trackers[2],output,output_feat,3,True,))
    t4 = threading.Thread(target=run_process, args=(sess2,l4_v,fs[3],count_human_lasts[3],metrics[3],trackers[3],output,output_feat,4,False,)) 

    # starting thread 1 
    t1.start() 
    # starting thread 2 
    t2.start() 
    # starting thread 3 
    t3.start() 
    # starting thread 4
    t4.start() 

    
    # wait until thread 1 is completely executed 
    t1.join() 
    # wait until thread 2 is completely executed 
    t2.join() 
    # wait until thread 3 is completely executed 
    t3.join() 
    # wait until thread 4 is completely executed 
    t4.join() 

    # reid cam1&2
    results = [output.get() for p in range(output.qsize())]
    results_feat = [output_feat.get() for p in range(output_feat.qsize())]

    results.sort(key=sortFirst)  
    results = [r[1] for r in results]
    results_feat.sort(key=sortFirst)  
    results_feat = [r[1] for r in results_feat]

    results0 = pd.DataFrame(results[1])
    results1 = pd.DataFrame(results[0])
    results2 = pd.DataFrame(results[2])
    results3 = pd.DataFrame(results[3])
    
    results_feat0 = pd.DataFrame(results_feat[1])
    results_feat1 = pd.DataFrame(results_feat[0])
    results_feat2 = pd.DataFrame(results_feat[2])
    results_feat3 = pd.DataFrame(results_feat[3])

    cam_0 = pd.concat([results0, results_feat0], axis=1)
    cam_0.columns = range(cam_0.shape[1])
    cam_1 = pd.concat([results1, results_feat1], axis=1)
    cam_1.columns = range(cam_1.shape[1])
    cam_2 = pd.concat([results2, results_feat2], axis=1)
    cam_2.columns = range(cam_2.shape[1])
    cam_3 = pd.concat([results3, results_feat3], axis=1)
    cam_3.columns = range(cam_3.shape[1])

    scope_cam_1 = cam_1[cam_1[3].between(200, 625)]
    scope_cam_0 = cam_0[cam_0[3].between(200, 625)] 
    scope_cam_2 = cam_2[cam_2[2].between(350, 750)]
    scope_cam_3 = cam_3

    # reid from previous video
    temp_reid_dict = reid_dict.copy()
    temp_reid_dict_2 = reid_dict_2.copy()

    #reid list
    reid_list = []

    if len(temp_reid_dict) != 0 or len(temp_reid_dict_2) != 0 :
        if len(temp_reid_dict) != 0 :
            for prev_id in temp_reid_dict :
                if prev_id in cam_1[1].unique() :
                    cam_1[1] = cam_1[1].replace(prev_id, reid_dict[prev_id])
                else :
                    if reid_life > 5 :
                        del reid_dict[prev_id]

        if len(temp_reid_dict_2) != 0 :
            for prev_id in temp_reid_dict_2 :
                if prev_id in cam_1[1].unique() or prev_id in cam_0[1].unique() or prev_id in cam_3[1].unique():
                    if prev_id in cam_1[1].unique() :
                        cam_1[1] = cam_1[1].replace(prev_id, reid_dict_2[prev_id])
                    if prev_id in cam_0[1].unique() :
                        cam_0[1] = cam_0[1].replace(prev_id, reid_dict_2[prev_id])
                    if prev_id in cam_3[1].unique() :
                        cam_3[1] = cam_3[1].replace(prev_id, reid_dict_2[prev_id])
                else :
                    if reid_life > 5 :
                        del reid_dict_2[prev_id]
        if reid_life > 5 :
            reid_life = 0

    done_check_1 = []
    done_check_2 = []

    re_cam_1 = scope_cam_1.copy()
    re_cam_2 = scope_cam_2.copy()
    
    #First

    for id_uni in scope_cam_1[1].unique():
        min_dis = [1000,'']
        for index1, row1 in tqdm(re_cam_1[re_cam_1[1]==id_uni].iterrows()):
            for index0, row0 in scope_cam_0.iterrows(): 
                dis = distance.cosine(np.array(row1.iloc[16:]).astype(float),np.array(row0.iloc[16:]).astype(float))
                if dis < min_dis[0] :
                    min_dis[0] = dis
                    min_dis[1] = row0[1]
        if min_dis[0] < 0.45 :
            print((id_uni, min_dis[1],min_dis[0]))
            # update result
            reid_dict[id_uni] = min_dis[1]
            cam_1[1] = cam_1[1].replace(id_uni, min_dis[1])
            reid_list.append([id_uni, min_dis[1]])
            
    #Second      

    scope_cam_1 = cam_1[cam_1[3].between(200, 625)]
    scope_cam_0 = cam_0[cam_0[3].between(200, 625)] 
    re_cam_1 = scope_cam_1.copy()
    re_cam_0 = scope_cam_0.copy()

    for id_uni in scope_cam_1[1].unique() :
        min_dis = [1000,'']
        for index1, row1 in tqdm(re_cam_1[re_cam_1[1]==id_uni].iterrows()):
            for index2, row2 in scope_cam_2.iterrows(): 
                dis = distance.cosine(np.array(row1.iloc[16:]).astype(float),np.array(row2.iloc[16:]).astype(float))
                if dis < min_dis[0] :
                    min_dis[0] = dis
                    min_dis[1] = row2[1]
        if min_dis[0] < 0.45 :
            print((id_uni, min_dis[1],min_dis[0]))
            # update result
            reid_dict_2[id_uni] = min_dis[1]
            cam_1[1] = cam_1[1].replace(id_uni, min_dis[1])
            cam_0[1] = cam_0[1].replace(id_uni, min_dis[1])
            reid_list.append([id_uni, min_dis[1]])

    scope_cam_1 = cam_1[cam_1[3].between(200, 625)]
    scope_cam_0 = cam_0[cam_0[3].between(200, 625)] 
    re_cam_1 = scope_cam_1.copy()
    re_cam_0 = scope_cam_0.copy()

    for id_uni in scope_cam_0[1].unique() :
        min_dis = [1000,'']
        for index0, row0 in tqdm(re_cam_0[re_cam_0[1]==id_uni].iterrows()):
            for index2, row2 in scope_cam_2.iterrows(): 
                dis = distance.cosine(np.array(row0.iloc[16:]).astype(float),np.array(row2.iloc[16:]).astype(float))
                if dis < min_dis[0] :
                    min_dis[0] = dis
                    min_dis[1] = row2[1]
        if min_dis[0] < 0.45 :
            print((id_uni, min_dis[1],min_dis[0]))
            # update result
            reid_dict_2[id_uni] = min_dis[1]
            cam_1[1] = cam_1[1].replace(id_uni, min_dis[1])
            cam_0[1] = cam_0[1].replace(id_uni, min_dis[1])
            reid_list.append([id_uni, min_dis[1]])

    scope_cam_3 = cam_3
    re_cam_3 = scope_cam_3.copy()

    for id_uni in scope_cam_3[1].unique() :
        min_dis = [1000,'']
        for index0, row0 in tqdm(re_cam_3[re_cam_3[1]==id_uni].iterrows()):
            for index2, row2 in scope_cam_2.iterrows(): 
                dis = distance.cosine(np.array(row0.iloc[16:]).astype(float),np.array(row2.iloc[16:]).astype(float))
                if dis < min_dis[0] :
                    min_dis[0] = dis
                    min_dis[1] = row2[1]
        print('/////////////////////////////////////////////////////////////////')
        print(min_dis[0])
        if min_dis[0] < 0.33:
            print((id_uni, min_dis[1],min_dis[0]))
            # update result
            reid_dict_2[id_uni] = min_dis[1]
            cam_3[1] = cam_3[1].replace(id_uni, min_dis[1])
            reid_list.append([id_uni, min_dis[1]])

    cam_0 = cam_0.iloc[:,:16]
    cam_1 = cam_1.iloc[:,:16]
    cam_2 = cam_2.iloc[:,:16]
    cam_3 = cam_3.iloc[:,:16]
    

    ############# Mapping #############

    Real_position = pd.DataFrame(data= {'user_id': [], 'block_id': [],'area_name': [], 'cam_id': [], 'video_name': [], 'date': [], 'time': [],'x': [],'y': [],'width': [],'height': []})
    
    Real_position  = assign_RealPosition(cam_0,Box_set,1,'120_',Real_position)
    # send 
    Real_position  = assign_RealPosition(cam_1,Box_set,2,'121_',Real_position)
    # send 
    Real_position  = assign_RealPosition(cam_2,Box_set,5,'220_',Real_position)
    # send 
    Real_position = assign_RealPosition(cam_3,Box_set,3,'122_',Real_position)
    # send 
    


    #Face reid 
    real_id_reid = []
    face_list = cam_2[cam_2[12].between(100,350) & cam_2[13].between(130,350) & cam_2[2].between(300, 750)][[0,1,10,11,12,13]]
    temp_face_embedding , face_pic_list  = get_face_embedding(video_dir,l3_v,face_list)
    
    if not new_face.empty() :
        while not new_face.empty():
            temp_nf = new_face.get()
            temp_nf = pd.concat([temp_nf[0],temp_nf.drop(columns=[0]).astype(float)],axis=1)
            face_embedding = pd.concat([face_embedding,temp_nf]).reset_index(drop=True)

    if len(temp_face_embedding) != 0 and len(face_embedding) != 0:
        temp_face_embedding = pd.concat([temp_face_embedding[0],temp_face_embedding.drop(columns=[0]).astype(float)],axis=1)
        face_embedding = pd.concat([face_embedding[0],face_embedding.drop(columns=[0]).astype(float)],axis=1)
        
        for uni_id in temp_face_embedding[0].unique():
            mindis = 10000
            real_id  = ''
            for index, row in temp_face_embedding[temp_face_embedding[0]==uni_id].iterrows():
                for index2, row2 in face_embedding.iterrows():
                    d = distance.euclidean(list(row[1:]), list(row2[1:]))
                    if d < mindis and d != 0:
                        mindis = d 
                        real_id = str(row2[0])
                        idx_face = index

            if mindis < 0.85 :


                print((row[0], real_id,mindis))
                #send reid 
                Real_position['user_id'] = Real_position['user_id'].replace(uni_id, real_id)
                reid_face_dict[uni_id] = real_id

                url = api_path+'/search/update'
                reid_data = {"old_id":uni_id,"new_id":real_id}
                x = requests.post(url, json = reid_data)

                cam_0[1] = cam_0[1].replace(uni_id, real_id)
                cam_1[1] = cam_1[1].replace(uni_id, real_id)
                cam_2[1] = cam_2[1].replace(uni_id, real_id)
                cam_3[1] = cam_3[1].replace(uni_id, real_id)
            else :

                face_pic = face_pic_list[face_pic_list[0]==uni_id].iloc[[idx_face]][1].to_list()[0]
                real_id = str('99_'+str(unknown_id))
                nameF = 'reid_face/'+str(real_id)+'.jpg'
                cv2.imwrite(nameF,face_pic)
                
                with open(nameF, "rb") as img_file:
                    my_string = base64.b64encode(img_file.read())

                url = api_path+'/search/unknown'
                unk_pic = {"user_id":real_id,"picture":my_string.decode('utf-8')}
                x = requests.post(url, json = unk_pic)

                print((uni_id, real_id))
                Real_position['user_id'] = Real_position['user_id'].replace(uni_id, real_id)
                reid_face_dict[uni_id] = real_id

                url = api_path+'/search/update'
                reid_data = {"old_id":uni_id,"new_id":real_id}
                x = requests.post(url, json = reid_data)

                cam_0[1] = cam_0[1].replace(uni_id, real_id)
                cam_1[1] = cam_1[1].replace(uni_id, real_id)
                cam_2[1] = cam_2[1].replace(uni_id, real_id)
                cam_3[1] = cam_3[1].replace(uni_id, real_id)
                unknown_id += 1

    temp_reid_face_dict = reid_face_dict.copy()
    if len(temp_reid_face_dict) != 0 :
        for prev_id in temp_reid_face_dict :
            if prev_id in Real_position['user_id'].unique() :
                Real_position['user_id'] = Real_position['user_id'].replace(prev_id, temp_reid_face_dict[prev_id])
                cam_0[1] = cam_0[1].replace(prev_id, temp_reid_face_dict[prev_id])
                cam_1[1] = cam_1[1].replace(prev_id, temp_reid_face_dict[prev_id])
                cam_2[1] = cam_2[1].replace(prev_id, temp_reid_face_dict[prev_id])
                cam_3[1] = cam_3[1].replace(prev_id, temp_reid_face_dict[prev_id])
            else :
                if reid_life > 5 :
                    del reid_face_dict[prev_id]
    if reid_life > 5 :
        reid_life = 0

    reid_life += 1

sess2.close()
print("Done!") 



