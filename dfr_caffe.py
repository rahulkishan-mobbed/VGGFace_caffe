'''
Created on Oct 27, 2017

@author: rpalyam
'''

import numpy as np
import sys
import caffe
import scipy.spatial.distance as ssd

def centre_img(image,crop_dims):
    center = np.array(crop_dims) / 2.0
    crop = np.tile(center, (1, 2))[0] + np.concatenate([
        -crop_dims / 2.0,
        crop_dims / 2.0
    ])
    crop = crop.astype(int)
    crops = image[crop[0]:crop[2], crop[1]:crop[3], :]
    crop_flip = crops[:, ::-1, :]

    return crops,crop_flip

def get_scores(net, img):
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
#     transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    image1 = caffe.io.load_image(img)
    img1 = caffe.io.resize_image(image1, (256,256))
    img2 = caffe.io.resize_image(image1, (384,384))
    img3 = caffe.io.resize_image(image1, (512,512))
    cen1, cflip1 = centre_img(img1,np.array([224,224]))
    cen2, cflip2 = centre_img(img2,np.array([224,224]))
    cen3, cflip3 = centre_img(img3,np.array([224,224]))
    net.blobs['data'].reshape(6,3,224,224)
    
    net.blobs['data'].data[0,...] = transformer.preprocess('data', cen1)
    net.blobs['data'].data[1,...] = transformer.preprocess('data', cflip1)
    net.blobs['data'].data[2,...] = transformer.preprocess('data', cen2)
    net.blobs['data'].data[3,...] = transformer.preprocess('data', cflip2)
    net.blobs['data'].data[4,...] = transformer.preprocess('data', cen3)
    net.blobs['data'].data[5,...] = transformer.preprocess('data', cflip3)
     
    net.forward()
    caffe_ft = net.blobs['prob'].data[0]
    print np.argmax(caffe_ft)
    
    return net.blobs['fc7'].data

def compute_distance(net, img1, img2):
    id1 = get_scores(net, img1)
    id1 = np.mean(id1, axis=0)
    id1_norm = id1 / np.linalg.norm(id1)
    id2 = get_scores(net,img2)
    id2 = np.mean(id2, axis=0)
    
    id2_norm = id2 / np.linalg.norm(id2)
    comp_dist = ssd.braycurtis(id1_norm, id2_norm)
    print comp_dist
    dist_eucl = ssd.euclidean(id1_norm, id2_norm)
    dist_cosine = ssd.cosine(id1_norm, id2_norm)
    return comp_dist, dist_cosine, dist_eucl

if __name__ == '__main__':
    caffe_root = '/home/rpalyam/Downloads/caffe-master/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

    model_def = caffe_root + 'models/vgg/VGG_FACE_deploy.prototxt'
    model_weights = caffe_root + 'models/vgg/VGG_FACE.caffemodel'

    net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
    
    faces_db = '/vol/corpora/faces/LFW/lfw-facedb/original_faces/'
    distances = []
    distances_c = []
    distances_e = []
    
    pairs_file = '/home/rpalyam/Documents/Tutworks/test/src/pairs.txt'
    with open(pairs_file, 'r') as fl_pairs:
        lines = fl_pairs.readlines()
        for line in lines:
            content = line.split()
            print line
            if len(content) == 2:
                print '2'
            elif len(content) == 3:
                img1 = faces_db + content[0] +'_'+ format(int(content[1]),'04d') + '.jpg'
                img2 = faces_db + content[0] +'_'+ format(int(content[2]),'04d') + '.jpg'
                curr_dist1, curr2, curr3 = compute_distance(net, img1, img2)
                distances = np.append(distances, curr_dist1)
                distances_c = np.append(distances_c, curr2)
                distances_e = np.append(distances_e, curr3)
            elif len(content) == 4:
                img1 = faces_db + content[0] +'_'+ format(int(content[1]),'04d') + '.jpg'
                img2 = faces_db + content[2] +'_'+ format(int(content[3]),'04d') + '.jpg'
                curr_dist, curr2, curr3 = compute_distance(net, img1, img2)
                if curr_dist > 1.0: print curr_dist
                distances = np.append(distances, curr_dist)
                distances_c = np.append(distances_c, curr2)
                distances_e = np.append(distances_e, curr3)
    np.save('distances.npy', distances)
    np.save('dist_cosine.npy', distances_c)
    np.save('dist_eucl.npy', distances_e)
