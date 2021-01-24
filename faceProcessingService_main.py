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

from base64            import b64decode
from skimage           import color
from skimage.transform import resize
from skimage.draw      import rectangle, rectangle_perimeter
from skimage.io        import imsave
from simplejpeg        import decode_jpeg

from faceProcessingService_common import FaceDetectionModel, FaceRecognitionModel, load_data

from flask      import Flask, request, jsonify
app = Flask(__name__)

detectionModel          = None
recognitionModel        = None
face_embedding_s_packet = None

verbose          = False

def processImage( source_image, verbose ):
    source_height, source_width, source_depth = source_image.shape
    
    detection_s = detectionModel( source_image[np.newaxis,:,:,:] )[0]
    #Process detected faces
    distance_s_s = list()
    key_s_s = list()
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
            face_embedding_s  = recognitionModel( face_image_s, field )
            face_embedding_s /= norm(face_embedding_s, axis=1)
            
            dict_values = face_embedding_s_packet['dict_values']
            dict_keys   = face_embedding_s_packet['dict_keys'  ]
            pca_model   = face_embedding_s_packet['pca_model'  ]
            nn_model    = face_embedding_s_packet['nn_model'   ]
            
            face_embedding_s = pca_model.transform( face_embedding_s )
            distance_s_numpy, index_s_numpy = nn_model.kneighbors( face_embedding_s )
            distance_s_list , index_s_list  = distance_s_numpy[0].tolist(), index_s_numpy[0].tolist()
            key_s_list = [ dict_keys[i] for i in index_s_list]
            
            distance_s_s.append(distance_s_list)
            key_s_s     .append(key_s_list     )
        else:
            print(x_min, x_max, y_min, y_max)
        #if verbose :
        #    start=(y_min,x_min)
        #    end  =(y_max,x_max)
        #    rr, cc = rectangle_perimeter(start=start, end=end, shape=(source_height,source_width))
        #    
        #    source_image[rr,cc,:] = 255
        #    imsave(os.path.join("output","temp.jpg"), source_image)
    return distance_s_s, key_s_s



def parseParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_caffemodel'  , default="models/FB_front_2nd.caffemodel"       , type=str  )
    parser.add_argument('--detection_prototxt'    , default="models/FB_front_2nd.prototxt"         , type=str  )
    parser.add_argument('--recognition_caffemodel', default="models/res18_arcVGG_ST_Aff.caffemodel", type=str  )
    parser.add_argument('--recognition_prototxt'  , default="models/res18_arcVGG_ST_Aff.prototxt"  , type=str  )
    parser.add_argument('--embeddings_data_file'  , default="datasets/testMSCeleb.pkl"             , type=str  )
    parser.add_argument('--field'                 , default="fc1"                                  , type=str  )
    parser.add_argument('--alpha'                 , default=1.3                                    , type=float)
    
    parser.add_argument('--gpu_id'                , default=2       , type=int)
    parser.add_argument('--output'                , default="output", type=str)
    parser.add_argument("--verbose"               , action="store_false"      )
    
    args = parser.parse_args()
    
    return args.detection_prototxt, args.detection_caffemodel, args.recognition_prototxt, args.recognition_caffemodel, args.embeddings_data_file, args.field, args.alpha, args.gpu_id, args.output, args.verbose

ROOT_DIR   = pt.abspath(pt.dirname(__file__))
OUTPUT_DIR = 'output'

@app.route('/api/img', methods=['POST'])
def upload():
    image = request.json['img']
    if (not image or type(image) is not str or not image.startswith('data:image')):
        return jsonify({'result':'no "img" uploaded'}), 400
    image_id = request.json['id']
    if (not image_id):
        return jsonify({'result':'no "image_id" passed'}), 400
    encoded_image = image.split(',')[1]
    #with open(image_id+".png","wb") as f:
    #    f.write(b64decode(image))
    source_image = decode_jpeg( b64decode(encoded_image) )
    distance_s_s, key_s_s = processImage( source_image, verbose )
    
    return jsonify({'distance_s':distance_s_s,'key_s':key_s_s})

if __name__ == '__main__':
    detection_prototxt, detection_caffemodel, recognition_prototxt, recognition_caffemodel, embeddings_data_file, field, alpha, gpu_id, output, verbose = parseParameters()
    
    caffe.set_mode_cpu()
    #caffe.set_mode_gpu()
    #caffe.set_device( gpu_id )
    
    detectionModel   = FaceDetectionModel  ( detection_prototxt  , detection_caffemodel   ) 
    print("Loading face detection model ... completed")
    recognitionModel = FaceRecognitionModel( recognition_prototxt, recognition_caffemodel ) 
    print("Loading face recognition model ... completed")
    face_embedding_s_packet = load_data( 'temporary.pkl' )
    print("Loading preprocessed face embeddings ... completed")
    
    app.run(host='0.0.0.0',port=5000)
