#!/usr/bin/python3 
import os.path as pt
import numpy   as np

import simplejpeg
import argparse
import pickle
import glob
import math
import os

os.environ['GLOG_minloglevel'] = '2'

import caffe

from base64            import b64decode
from skimage           import color
from skimage.transform import resize
from skimage.draw      import rectangle_perimeter
from skimage.io        import imsave

detectionModel = None
recognitionModel = None

class FaceDetectionModel( object ):
    def __init__(self, proto_filename, weight_filename ):
        self.model        = caffe.Net( proto_filename, weight_filename, caffe.TEST )
    def __call__( self, source_image_s ):
        batch, height, width, depth = source_image_s.shape
        
        source_image_s = np.transpose( source_image_s, axes=(0,3,1,2) )
        source_image_s = source_image_s[:,[2,1,0],:,:]
        source_image_s = source_image_s.astype(np.float)/255
        
        self.model.blobs['data'].reshape( *source_image_s.shape )
        self.model.blobs['data'].data[...] = source_image_s
        self.model.forward()
        
        detection_s_numpy = self.model.blobs['detection_s'].data[...]
        size_s_numpy      = self.model.blobs['size_s'     ].data[...]
        
        detection_s_list = [None]*batch
        for i in range( batch ):
            size = int( size_s_numpy[i] )
            detection_s_list[i] = detection_s_numpy[ i, :size, : ]
        return detection_s_list

class FaceRecognitionModel( object ):
    def __init__(self, proto_filename, weight_filename ):
        self.model        = caffe.Net( proto_filename, weight_filename, caffe.TEST )
        self.input_shape  = self.model.blobs['data'].data.shape
    def __call__( self, source_image_s, field ):
        target_image_s = np.zeros((source_image_s.shape[0],224,224,3))
        for n, source_image in enumerate(source_image_s):
            target_image_s[n,:,:,:] = resize(source_image[:,:,:], (224,224), order=1)
        target_image_s = np.transpose( target_image_s, axes=(0,3,1,2) )
        target_image_s = target_image_s[:,[2,1,0],:,:]
        
        self.model.blobs['data'].reshape( *target_image_s.shape )
        self.model.blobs['data'].data[...] = target_image_s
        self.model.forward()
        embedding_s = self.model.blobs[field]
        return np.copy( embedding_s.data )

def load_data_from_pickle(name ):
    with open( name , 'rb') as f:
        return pickle.load(f)
