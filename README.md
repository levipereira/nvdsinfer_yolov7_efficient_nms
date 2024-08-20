# NvDsInferYolov7EfficientNMS for Gst-nvinferserver

# This repo is being deprecated: Please use https://github.com/levipereira/nvdsinfer_yolo

This repository provides a custom implementation of parsing function to the Gst-nvinferserver (NVIDIA DEEPSTREAM) plugin when use YOLOv7 model served by Triton Server using the [Efficient NMS](https://github.com/NVIDIA/TensorRT/tree/master/plugin/efficientNMSPlugin) plugin exported by ONNX.


By using the parsing function provided by NvDsInferYolov7EfficientNMS, handling the number of classes dynamically becomes easier. This eliminates the need to hardcode the number of classes, allowing the same plugin to be used for different YOLOv7 models with varying numbers of classes.


# Deployment Guide for NvDsInferYolov7EfficientNMS

## Cloning Repository and Installation

To clone the repository and install the NvDsInferYolov7EfficientNMS library, follow these steps:


```bash
# Clone the repository
git clone https://github.com/levipereira/nvdsinfer_yolov7_efficient_nms.git

# Copy the repository to the desired location
cp -R nvdsinfer_yolov7_efficient_nms/ /opt/nvidia/deepstream/deepstream/sources/libs/

# Set the CUDA_VER environment variable (check your deepstream cuda version.  The DS 6.4 use cuda 12.2)
export CUDA_VER=12.2

# Navigate to the directory containing the nvdsinfer_yolov7_efficient_nms library
cd /opt/nvidia/deepstream/deepstream/sources/libs/nvdsinfer_yolov7_efficient_nms

# Build the project using the provided MakeFile
make -f MakeFile all
make -f MakeFile install

```
Install Location:

`/opt/nvidia/deepstream/deepstream/lib/libnvds_infer_yolov7_efficient_nms.so`

Usage on Deepstream

Snippet [Gst-nvinferserver](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html)  Configuration File
```
  postprocess {
    labelfile_path: "labels.txt"
    detection {
      num_detected_classes: 80
      custom_parse_bbox_func: "NvDsInferYolov7EfficientNMS"
    }
  }
  custom_lib {
    path : "/opt/nvidia/deepstream/deepstream-6.4/lib/libnvds_infer_yolov7_efficient_nms.so"
  }
```

