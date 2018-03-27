import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import tempfile
from caffe import layers as L
from caffe import params as P
import os
import sys
import cv2
import caffe
from caffe.proto import caffe_pb2

vgg_face_model = '/home/rpalyam/Downloads/caffe-master/models/vgg/'
vgg_new_model = '/data/rpalyam/VGGmodels/'
vgg_lmdb = '/data/rpalyam/VGGdataset_dfi'
vgg_test_lmdb = '/data/rpalyam/VGG_test_dfi'

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=1, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2
transform_params = dict(scale=1, crop_size=224, mirror=1)

if os.path.isfile(vgg_face_model + 'VGG_FACE.caffemodel'):
    weights = vgg_face_model + 'VGG_FACE.caffemodel'
    
def deprocess_image(imagen):
    image = imagen.copy()              # don't modify destructively
    image = image[0,:,:,:]               # Using just CHW 
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    print image.shape

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.01),
            bias_filler=dict(type='constant', value=0)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)
    
def caffenet(data, label=None, train=True, num_classes=2622,
             classifier_name='fc8_dfi', learn_all=False):
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1_1 =  L.Convolution(n.data , kernel_size=3, stride=1,
                         num_output=64, pad=1, param=param)
    n.relu1_1 = L.ReLU(n.conv1_1, in_place=True)
    
    n.conv1_2 =  L.Convolution(n.relu1_1, kernel_size=3, stride=1,
                         num_output=64, pad=1, param=param)
    n.relu1_2 = L.ReLU(n.conv1_2, in_place=True)
    
    n.pool1 =L.Pooling(n.relu1_2, pool=P.Pooling.MAX, 
                       kernel_size=2, stride=2)
    
    n.conv2_1 =  L.Convolution(n.pool1, kernel_size=3, stride=1,
                         num_output=128, pad=1, param=param)
    n.relu2_1 = L.ReLU(n.conv2_1, in_place=True)
    
    n.conv2_2 =  L.Convolution(n.relu2_1, kernel_size=3, stride=1,
                         num_output=128, pad=1, param=param)
    n.relu2_2 = L.ReLU(n.conv2_2, in_place=True)
    
    n.pool2 =L.Pooling(n.relu2_2, pool=P.Pooling.MAX, 
                       kernel_size=2, stride=2)
    
    n.conv3_1 =  L.Convolution(n.pool2, kernel_size=3, stride=1,
                         num_output=256, pad=1, param=param)
    n.relu3_1 = L.ReLU(n.conv3_1, in_place=True)
    
    n.conv3_2 =  L.Convolution(n.relu3_1, kernel_size=3, stride=1,
                         num_output=256, pad=1, param=param)
    n.relu3_2 = L.ReLU(n.conv3_2, in_place=True)
    
    n.conv3_3 =  L.Convolution(n.relu3_2, kernel_size=3, stride=1,
                         num_output=256, pad=1, param=param)
    n.relu3_3 = L.ReLU(n.conv3_3, in_place=True)
    
    n.pool3 =L.Pooling(n.relu3_3, pool=P.Pooling.MAX, 
                       kernel_size=2, stride=2)
    
    n.conv4_1 =  L.Convolution(n.pool3, kernel_size=3, stride=1,
                         num_output=512, pad=1, param=param)
    n.relu4_1 = L.ReLU(n.conv4_1, in_place=True)
    
    n.conv4_2 =  L.Convolution(n.relu4_1, kernel_size=3, stride=1,
                         num_output=512, pad=1, param=param)
    n.relu4_2 = L.ReLU(n.conv4_2, in_place=True)
    
    n.conv4_3 =  L.Convolution(n.relu4_2, kernel_size=3, stride=1,
                         num_output=512, pad=1, param=param)
    n.relu4_3 = L.ReLU(n.conv4_3, in_place=True)
    
    n.pool4 =L.Pooling(n.relu4_3, pool=P.Pooling.MAX, 
                       kernel_size=2, stride=2)

    n.conv5_1 =  L.Convolution(n.pool4, kernel_size=3, stride=1,
                         num_output=512, pad=1, param=param)
    n.relu5_1 = L.ReLU(n.conv5_1, in_place=True)
    
    n.conv5_2 =  L.Convolution(n.relu5_1, kernel_size=3, stride=1,
                         num_output=512, pad=1, param=param)
    n.relu5_2 = L.ReLU(n.conv5_2, in_place=True)
    
    n.conv5_3 =  L.Convolution(n.relu5_2, kernel_size=3, stride=1,
                         num_output=512, pad=1, param=param)
    n.relu5_3 = L.ReLU(n.conv5_3, in_place=True)
    
    n.pool5 =L.Pooling(n.relu5_3, pool=P.Pooling.MAX, 
                       kernel_size=2, stride=2)

    n.fc6_dfi, n.relu6_dfi = fc_relu(n.pool5, 4096, param=learned_param)
    if train:
        n.drop6_dfi = fc7input = L.Dropout(n.relu6_dfi, in_place=True)
    else:
        fc7input = n.relu6_dfi
    n.fc7_dfi, n.relu7_dfi = fc_relu(fc7input, 4096, param=learned_param)
    if train:
        n.drop7_dfi = fc8input = L.Dropout(n.relu7_dfi, in_place=True)
    else:
        fc8input = n.relu7_dfi
    # always learn fc8 (param=learned_param)
    fc8_dfi = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)

    n.__setattr__(classifier_name, fc8_dfi)
    if not train:
        n.prob = L.Softmax(fc8_dfi)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8_dfi, n.label)
        n.acc = L.Accuracy(fc8_dfi, n.label)
    
    return n.to_proto()
   


def vggnet(source, label=None, train=True, subset=None,
               num_classes=1000, learn_all = False):
    if subset is None:
        subset = 'train' if train else 'test'
    if train: 
        transform_param = dict(mirror= 1 , crop_size = 224)
        vgg_data, vgg_label= L.Data(batch_size=64, backend=P.Data.LMDB, source=source, ntop=2,
                                transform_param=transform_param)  
    else:
                vgg_data, vgg_label= L.Data(batch_size=64, backend=P.Data.LMDB, source=source, ntop=2)
        
    return caffenet(data=vgg_data, label=vgg_label, train=train,
                     num_classes=num_classes, learn_all=learn_all)


def vgg_net_proto():
    with open(vgg_new_model + 'VGG_dfi_train.prototxt' , 'w') as f:
        f.write(str(vggnet(source= vgg_lmdb, 
                               train=True, subset='train',
                               num_classes=2622)  ))

    with open(vgg_new_model + 'VGG_dfi_test.prototxt' , 'w') as f:
        f.write(str(vggnet(source= vgg_test_lmdb, 
                               train=False, subset='test',
                               num_classes=2622)  ))
    
    return  


def solver(train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 5000  # Test after every 1000 training iterations.
        s.test_iter.append(64) # Test on 64 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 256
    
    s.max_iter = 100000     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 8000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 10000
    s.snapshot_prefix = vgg_new_model + 'VGG_ft_dfi'
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
    with open(vgg_new_model + 'VGG_dfi_solver.prototxt' , 'w') as f:
        f.write(str(s))
    
    return

def run_solver(niter = 18000, disp_interval = 10):
    solver = caffe.get_solver(vgg_new_model + 'VGG_dfi_solver.prototxt')
    print 'Solver loaded'
    
    if os.path.isfile(vgg_face_model + 'VGG_FACE.caffemodel'):
        weights = vgg_face_model + 'VGG_FACE.caffemodel'
        print 'Caffe VGG weights found'
        
    
    solver.net.copy_from(weights)
    print 'weights copied'
    
    loss = np.zeros(niter)
    acc = np.zeros(niter)
    for it in range(niter):
        solver.step(1) #Run a single SGD step # Simulate a batch size of 
        loss[it] = solver.net.blobs['loss'].data.copy()
        acc[it] = solver.net.blobs['acc'].data.copy()

        if it % disp_interval == 0 or it + 1 == niter:

            loss_disp = 'Loss:', loss[it], ' Acc:', np.round(100*acc[it])
            print '%3d) %s' % (it, loss_disp)
            
 
    # Save the learned weights.
    weights = os.path.join(vgg_new_model, 'weights.VGG_dfi.caffemodel')
    solver.net.save(weights)           
    
    return


if __name__ == '__main__':

    model_train = vgg_new_model + 'VGG_dfi_train.prototxt'
    
    model_test = vgg_new_model + 'VGG_dfi_test.prototxt'

#     solver(model_train, model_test, base_lr=0.001)

    caffe.set_device(0)
#     vgg_net_proto()

    run_solver()        
    print 'Done'
