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

from sklearn.neighbors     import NearestNeighbors
from sklearn.decomposition import PCA

from numpy.linalg          import norm

from skimage.transform     import resize
from skimage.draw          import rectangle, rectangle_perimeter
from skimage.io            import imsave

from simplejpeg            import decode_jpeg

from tqdm                  import tqdm

from faceProcessingService_common import FaceDetectionModel, FaceRecognitionModel, save_data

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

def processInputDataset( input_dataset_dir ):
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
    face_embedding_s_numpy = np.array( list(face_embedding_s_dict.values()))
    pca_model = PCA(n_components=0.95)
    pca_model.fit( face_embedding_s_numpy )
    
    face_embedding_s_dict   = { key: pca_model.transform( value.reshape( 1,-1))[0] for key, value in face_embedding_s_dict.items()}
    face_embedding_s_dict   = { key: value/norm(value)                             for key, value in face_embedding_s_dict.items()}
    face_embedding_s_keys   = list( face_embedding_s_dict.keys() )
    face_embedding_s_values = np.array( list(face_embedding_s_dict.values()) )
    
    nn_model = NearestNeighbors( n_neighbors=10, algorithm='ball_tree')
    nn_model.fit( face_embedding_s_values )
    
    face_embedding_s_packet = dict()
    face_embedding_s_packet['dict_values'] = face_embedding_s_values
    face_embedding_s_packet['dict_keys'  ] = face_embedding_s_keys
    face_embedding_s_packet['pca_model'  ] = pca_model
    face_embedding_s_packet['nn_model'   ] = nn_model
    
    return face_embedding_s_packet


if __name__ == '__main__':
    detection_prototxt, detection_caffemodel, recognition_prototxt, recognition_caffemodel, input_dataset_dir, field, alpha, gpu_id, output_file, verbose = parseParameters()
    
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device( gpu_id )
    
    detectionModel   = FaceDetectionModel  ( detection_prototxt  , detection_caffemodel   ) 
    print("Loading face detection model ... completed")
    recognitionModel = FaceRecognitionModel( recognition_prototxt, recognition_caffemodel ) 
    print("Loading face recognition model ... completed")
    
    faceEmbeddingsPacket = processInputDataset( input_dataset_dir )

    save_data( faceEmbeddingsPacket, f"temporary.pkl")
