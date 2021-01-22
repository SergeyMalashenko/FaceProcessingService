#CUDA 10.2
#export CUDA_HOME=/usr/local/cuda-10.2
#export PATH=${PATH}:${CUDA_HOME}/bin
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
#OpenCV
#export OPENCV_HOME=/usr/local/opencv-3.4.10_cuda-10.2
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${OPENCV_HOME}/lib
#export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${OPENCV_HOME}/lib/pkgconfig
#export PYTHONPATH=${OPENCV_HOME}/lib/python2.7/dist-packages:${PYTHONPATH}
#Caffe
export CAFFE_ROOT=~/caffe/build_release/install 
#export CAFFE_ROOT=`pwd`/build_debug/install 
export LD_LIBRARY_PATH=${CAFFE_ROOT}/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${CAFFE_ROOT}/python:${PYTHONPATH}


