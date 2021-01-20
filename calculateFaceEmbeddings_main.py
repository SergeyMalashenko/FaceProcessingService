#!/usr/bin/python3
import os.path as pt
import numpy   as np
import scipy   as sp

import simplejpeg
import argparse
import pickle
import glob
import math
import os

os.environ['GLOG_minloglevel'] = '2'

import caffe

from skimage.transform import resize
from skimage.draw      import rectangle, rectangle_perimeter
from skimage.io        import imsave
from simplejpeg        import decode_jpeg

from tqdm              import tqdm

from faceProcessingService_common import FaceDetectionModel, FaceRecognitionModel

detectionModel   = None
recognitionModel = None
verbose          = False

def parseParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_caffemodel'  , default="models/FB_front_2nd.caffemodel"       , type=str  )
    parser.add_argument('--detection_prototxt'    , default="models/FB_front_2nd.prototxt"         , type=str  )
    parser.add_argument('--recognition_caffemodel', default="models/res18_arcVGG_ST_Aff.caffemodel", type=str  )
    parser.add_argument('--recognition_prototxt'  , default="models/res18_arcVGG_ST_Aff.prototxt"  , type=str  )
    
    parser.add_argument('--input_dataset_dir'     , default="datasets"                             , type=str  )
    
    parser.add_argument('--field'                 , default="fc1"                                  , type=str  )
    parser.add_argument('--alpha'                 , default=1.3                                    , type=float)
    
    parser.add_argument('--gpu_id'                , default=2       , type=int)
    parser.add_argument('--output_file'           , default="output", type=str)
    parser.add_argument("--verbose"               , action="store_false"      )
    
    args = parser.parse_args()
    
    return args.detection_prototxt, args.detection_caffemodel, args.recognition_prototxt, args.recognition_caffemodel, args.input_dataset_dir, args.field, args.alpha, args.gpu_id, args.output_file, args.verbose

ROOT_DIR   = pt.abspath(pt.dirname(__file__))

def get( source_image, verbose ):
    source_height, source_width, source_depth = source_image.shape
    
    detection_s = detectionModel( source_image[np.newaxis,:,:,:] )[0]
    #Process detected faces
    for detection in detection_s:
        x_min, y_min, x_max, y_max = detection[2], detection[3], detection[4], detection[5]
        x_center, y_center = (x_min + x_max)*  0.5*source_width, (y_min + y_max)*  0.5*source_height
        length = max( (x_max - x_min)*alpha*source_width, (y_max - y_min)*alpha*source_height )
        
        x_min = math.floor(x_center - length/2)
        x_max = math.ceil (x_center + length/2)
        y_min = math.floor(y_center - length/2)
        y_max = math.ceil (y_center + length/2)
        
        if (0 <= x_min ) and (x_max < source_width) and (0 <= y_min ) and (y_max < source_height) :
            face_image_s     = source_image[np.newaxis,(y_min+0):(y_max+1), (x_min+0):(x_max+1),:]
            face_embedding_s = recognitionModel( face_image_s, field )
            print(face_embedding_s)
        else:
            print(x_min, x_max, y_min, y_max)
        if verbose :
            start=(y_min,x_min)
            end  =(y_max,x_max)
            rr, cc = rectangle_perimeter(start=start, end=end, shape=(source_height,source_width))
            
            source_image[rr,cc,:] = 255
            imsave(os.path.join("output","temp.jpg"), source_image)
    return None



if __name__ == '__main__':
    detection_prototxt, detection_caffemodel, recognition_prototxt, recognition_caffemodel, input_dataset_dir, field, alpha, gpu_id, output_file, verbose = parseParameters()
    
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device( gpu_id )
    
    detectionModel   = FaceDetectionModel  ( detection_prototxt  , detection_caffemodel   ) 
    print("Loading face detection model ... completed")
    recognitionModel = FaceRecognitionModel( recognition_prototxt, recognition_caffemodel ) 
    print("Loading face recognition model ... completed")
    
    face_embedding_s_dict = dict()
    for subdir, dirs, files in tqdm( os.walk( input_dataset_dir ) ):
        for file in files:
            #Process images
            if file.endswith(".jpg"):
                current_file = os.path.join(subdir, file)
                with open(current_file, 'rb') as fp:
                    source_image = decode_jpeg( fp.read() )
                    source_height, source_width, source_depth = source_image.shape
                    
                    detection_s = detectionModel( source_image[np.newaxis,:,:,:] )[0]
                    #Process detected faces
                    for detection in detection_s:
                        x_min, y_min, x_max, y_max = detection[2], detection[3], detection[4], detection[5]
                        x_center, y_center = (x_min + x_max)*  0.5*source_width, (y_min + y_max)*  0.5*source_height
                        length = max( (x_max - x_min)*alpha*source_width, (y_max - y_min)*alpha*source_height )
                        
                        x_min = math.floor(x_center - length/2)
                        x_max = math.ceil (x_center + length/2)
                        y_min = math.floor(y_center - length/2)
                        y_max = math.ceil (y_center + length/2)
                        
                        if (0 <= x_min ) and (x_max < source_width) and (0 <= y_min ) and (y_max < source_height) :
                            face_embedding_s_dict[current_file] = recognitionModel( source_image[np.newaxis,(y_min+0):(y_max+1), (x_min+0):(x_max+1),:], field )[0]
    print(face_embedding_s_dict)

