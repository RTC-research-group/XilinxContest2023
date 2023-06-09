import cv2
import torchvision
from PIL import Image
import numpy as np
from typing import List
from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects
from trt_pose.coco import coco_category_to_topology
import argparse

# https://github.com/Hematies/trt_pose/blob/master/trt_pose/coco.py
preprocessingTransforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

poseTargetCategory = None # ... es un diccionario que no sé cómo se rellena. Lo suyo es poner una persona como categoría:
'''
 with open(annotations_file, 'r') as f:
            data = json.load(f)

        person = [c for c in data['categories'] if c['name'] == "person"][0]
'''

topology = coco_category_to_topology(poseTargetCategory)

# https://github.com/Hematies/trt_pose/blob/master/trt_pose/parse_objects.py#L4
postProcessing = ParseObjects(topology, cmap_threshold=0.1, link_threshold=0.1, cmap_window=5, line_integral_samples=7,
                             max_num_parts=100, max_num_objects=100)

# https://github.com/Hematies/trt_pose/blob/master/trt_pose/draw_objects.py
drawing = DrawObjects(topology)

def postProcessBatch(batch, originalImgs):
    res = []
    for k in range(batch.shape[0]):
        cmap, paf = batch[k]
        object_counts, objects, peaks = postProcessing(cmap, paf)
        object_counts, objects, peaks = int(object_counts[0]), objects[0], peaks[0]

        res.append(drawing(originalImgs[k], object_counts, objects, peaks))
    return np.array(res)

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def runDPU(inputBatch, dpuRunner):
    '''get tensor'''
    inputTensors = dpuRunner.get_input_tensors()
    outputTensors = dpuRunner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    # we can avoid output scaling if use argmax instead of softmax
    # output_fixpos = outputTensors[0].get_attr("fix_point")
    # output_scale = 1 / (2**output_fixpos)

    batchSize = input_ndim[0]
    n_of_images = len(inputBatch)
    count = 0
    ids = []
    ids_max = 10
    outputData = []
    res = [None] * n_of_images
    for i in range(ids_max):
        outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
    while count < n_of_images:
        if (count + batchSize <= n_of_images):
            runSize = batchSize
        else:
            runSize = n_of_images - count

        '''prepare batch input/output '''
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = inputBatch[(count + j) % n_of_images].reshape(input_ndim[1:])
        '''run with batch '''
        job_id = dpuRunner.execute_async(inputData, outputData[len(ids)])
        ids.append((job_id, runSize, count))
        count = count + runSize
        if count < n_of_images:
            if len(ids) < ids_max - 1:
                continue
        for index in range(len(ids)):
            dpuRunner.wait(ids[index][0])
            write_index = ids[index][2]
            '''store output vectors '''
            for j in range(ids[index][1]):
                res[write_index] = outputData[index][0][j]
                write_index += 1
        ids = []
    return np.numpy(res)


if __name__ == "__main__":
