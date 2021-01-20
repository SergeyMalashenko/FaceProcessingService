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
from skimage.transform import resize
from skimage.draw      import rectangle_perimeter
from skimage.io        import imsave

from faceProcessingCommon import FaceDetectionModel, FaceRecognitionModel

detectionModel = None
recognitionModel = None

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

ROOT_DIR = pt.abspath(pt.dirname(__file__))
OUTPUT_DIR = 'output'

if __name__ == '__main__':
    detection_prototxt, detection_caffemodel, recognition_prototxt, recognition_caffemodel, embeddings_data_file, field, alpha, gpu_id, output, verbose = parseParameters()
    
    caffe.set_mode_gpu()
    caffe.set_device( gpu_id )
    
    detectionModel   = FaceDetectionModel  ( detection_prototxt  , detection_caffemodel   ) 
    print("Loading face detection model ... completed")
    recognitionModel = FaceRecognitionModel( recognition_prototxt, recognition_caffemodel ) 
    print("Loading face recognition model ... completed")
    #reference_embedding_s_dict = load_data_from_pickle( embeddings_data_file )
    print("Loading preprocessed face embeddings ... completed")
    
    image_dir = pt.join(ROOT_DIR, 'data', '*.jpg')
    for image_path in glob.iglob(image_dir):
        #Load image
        with open(image_path, 'rb') as fp:
            plain_data   = fp.read()

        #Detect face
        decoded_image = simplejpeg.decode_jpeg(plain_data)
        input_height, input_width, input_depth = decoded_image.shape
        
        #pad_y = int( input_height*0.05 )
        #pad_x = int( input_width *0.05 )
        #processed_image = np.pad( decoded_image, ((pad_y,pad_y),(pad_x,pad_x),(0,0)), 'symmetric' )
        processed_image = decoded_image
        processed_height, processed_width, processed_depth = processed_image.shape
        
        input_image_s = processed_image[np.newaxis,:,:,:]
        detection_s_s = detectionModel( input_image_s )
        #Process detected faces
        for input_image, detection_s in zip( input_image_s, detection_s_s):
            for detection in detection_s:
                x_min, y_min, x_max, y_max = detection[2], detection[3], detection[4], detection[5]
                x_center, y_center = (x_min + x_max)*  0.5*processed_width, (y_min + y_max)*  0.5*processed_height
                length = max( (x_max - x_min)*alpha*processed_width, (y_max - y_min)*alpha*processed_height )
                
                x_min = math.floor(x_center - length/2)
                x_max = math.ceil (x_center + length/2)
                y_min = math.floor(y_center - length/2)
                y_max = math.ceil (y_center + length/2)
                
                if (0 <= x_min ) and (x_max < processed_width) and (0 <= y_min ) and (y_max < processed_height) :
                    face_image_s     = input_image[np.newaxis,(y_min+0):(y_max+1), (x_min+0):(x_max+1),:]
                    face_embedding_s = recognitionModel( face_image_s, field )
                    print(face_embedding_s)
                else:
                    print(x_min, x_max, y_min, y_max)
                if verbose :
                    start=(y_min,x_min)
                    end  =(y_max,x_max)
                    rr, cc = rectangle_perimeter(start=start, end=end, shape=(processed_height,processed_width))
                    
                    input_image[rr,cc,:] = 255
                    image_filename = os.path.basename(image_path)
                    imsave(os.path.join("output",image_filename), input_image)
        #Process embedding
                
    #Output result


