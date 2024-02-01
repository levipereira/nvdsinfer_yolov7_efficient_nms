/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.



 * Modifications made by Levi Pereira (https://forums.developer.nvidia.com/u/levi_pereira/activity):
 *  - Customized the function NvDsInferParseCustomEfficientNMS by creating the NvDsInferYolov7EfficientNMS.
 */



#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cmath>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))


/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferYolov7EfficientNMS (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList);


extern "C"
bool NvDsInferYolov7EfficientNMS (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    int* p_keep_count = (int *) outputLayersInfo[3].buffer;


    float* p_bboxes = (float *) outputLayersInfo[0].buffer;

    NvDsInferDims inferDims_p_bboxes = outputLayersInfo[0].inferDims;

    int numElements_p_bboxes=inferDims_p_bboxes.numElements;

    float* p_scores = (float *) outputLayersInfo[2].buffer;

    unsigned int* p_classes = (unsigned int *) outputLayersInfo[1].buffer;


    const float threshold = detectionParams.perClassThreshold[0];

    float max_bbox=0;
    for (int i=0; i < numElements_p_bboxes; i++)
    {
        if ( max_bbox < p_bboxes[i] )
            max_bbox=p_bboxes[i];
    }

    if (p_keep_count[0] > 0)
    {
        assert (!(max_bbox < 2.0));
        for (int i = 0; i < p_keep_count[0]; i++) {

            if ( p_scores[i] < threshold) continue;
            if ((unsigned int) p_classes[i] >= detectionParams.numClassesConfigured) {
                printf("Error: The number of classes configured in the GIE config-file (postprocess > num_detected_classes) is incorrect.\n");
                printf("Detected class index: %u\n", (unsigned int) p_classes[i]);
            }
            assert((unsigned int) p_classes[i] < detectionParams.numClassesConfigured);

            NvDsInferObjectDetectionInfo object;
            object.classId = (int) p_classes[i];
            object.detectionConfidence = p_scores[i];

            object.left=p_bboxes[4*i];
            object.top=p_bboxes[4*i+1];
            object.width=(p_bboxes[4*i+2] - object.left);
            object.height= (p_bboxes[4*i+3] - object.top);

            if(log_enable != NULL && std::stoi(log_enable)) {
                std::cout << "label/conf/ x/y w/h -- "
                << p_classes[i] << " "
                << p_scores[i] << " "
                << object.left << " " << object.top << " " << object.width << " "<< object.height << " "
                << std::endl;
            }

            object.left=CLIP(object.left, 0, networkInfo.width - 1);
            object.top=CLIP(object.top, 0, networkInfo.height - 1);
            object.width=CLIP(object.width, 0, networkInfo.width - 1);
            object.height=CLIP(object.height, 0, networkInfo.height - 1);

            objectList.push_back(object);
        }
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferYolov7EfficientNMS);

