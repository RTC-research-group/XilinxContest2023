# [Xilinx Contest 2023](http://www.openhw.eu/)

## Quantizing & Compiling TRT_Pose model with Vitis-AI

1. Download pretrained model from [trt_pose github](https://github.com/NVIDIA-AI-IOT/trt_pose) &rarr; https://drive.google.com/open?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd

2. Download COCO validation dataset, 2017 with keypoints annotations
    
    * [Validation dataset](http://images.cocodataset.org/zips/val2017.zip)
    * [Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

3. Preprocess COCO annotations with vitis/calib_dataset/preprocess_coco_person.py 

    * ```python3 preprocess_coco_person.py <input_annotation_file> <output_annotation_file>```
    * Use `person_keypoints_val2017.json` as input file

4. Apply quantization with vitis/quantize_trt_pose.py
    
    * `python3 quantize_trt_pose.py --model_dir <pretrained_model.pth> --quant_mode calib`

[comment]: <> (TODO: revisar quantize_trt_pose.py para que los argumentos tengan sentido y se especifique el dataset de calibracion desde la llamada)

5. Compile model with vitis/vai_c_xir_cust_pytorch.py
    
    * `python3 vai_c_xir_cust_pytorch.py -x <xmodel> -a <arch> -o <output_dir>`
    * `<xmodel>` is the .xmodel file resulting from `quantize_trt_pose.py execution`
    * `<arch>` is a JSON file containing the DPU architecture. This is found in the Vitis-AI tool installation under the route `/opt/vitis_ai/compiler/arch/<DPU>/<HW_PLATFORM>/arch.json`

6. The result should be 3 files:
    
    * md5sum.txt
    * meta.json
    * <model_name>.xmodel