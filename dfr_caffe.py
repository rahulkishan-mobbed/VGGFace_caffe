'''
Created on Oct 27, 2017

@author: rpalyam
'''
import numpy as np
import sys
import caffe
import scipy.spatial.distance as ssd

def compute_distance(net, img1, img2):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
#     transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    image1 = caffe.io.load_image(img1)
    transformed_image = transformer.preprocess('data', image1)
     
    net.blobs['data'].data[...] = transformed_image
     
    net.forward()
    caffe_ft = net.blobs['prob'].data
    print np.argmax(caffe_ft)
     
    id1 = net.blobs['fc7'].data
    id1_norm = id1 / np.linalg.norm(id1)
     
    image2 = caffe.io.load_image(img2)
    transformed_image = transformer.preprocess('data', image2)
     
    net.blobs['data'].data[...] = transformed_image
     
    net.forward()
    caffe_ft = net.blobs['prob'].data
    print np.argmax(caffe_ft)
    id2 = net.blobs['fc7'].data
    id2_norm = id2 / np.linalg.norm(id2)
     
    comp_dist = ssd.cdist(id1_norm, id2_norm, 'braycurtis')
    dist_cosine = ssd.cdist(id1_norm, id2_norm, 'cosine')
    dist_eucl = ssd.cdist(id1_norm, id2_norm, 'euclidean')
    return comp_dist, dist_cosine, dist_eucl

if __name__ == '__main__':
    caffe_root = '/home/rpalyam/Downloads/caffe-master/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
    sys.path.insert(0, caffe_root + 'python')

    caffe.set_mode_cpu()

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
                distances_c = np.append(distances, curr2)
                distances_e = np.append(distances, curr3)
            elif len(content) == 4:
                img1 = faces_db + content[0] +'_'+ format(int(content[1]),'04d') + '.jpg'
                img2 = faces_db + content[2] +'_'+ format(int(content[3]),'04d') + '.jpg'
                curr_dist, curr2, curr3 = compute_distance(net, img1, img2)
                if curr_dist > 1.0: print curr_dist
                distances = np.append(distances, curr_dist)
                distances_c = np.append(distances, curr2)
                distances_e = np.append(distances, curr3)
#     print distances
#     
#     dist = np.copy(distances)
    np.save('distances.npy', distances)
    np.save('dist_cosine.npy', distances_c)
    np.save('dist_eucl.npy', distances_e)
