# Edge AI Software And Development Tools

## Notice
Our documentation landing pages are the following:
- https://www.ti.com/tinyml : Technology page summarizing TI’s edge AI software/hardware products 
- https://github.com/TexasInstruments/tinyml : Landing page for developers to understand overall software and tools offering  

## Introduction

Embedded inference of Deep Learning models is quite challenging - due to high compute requirements. TI’s Edge AI comprehensive software product help to optimize and accelerate inference on TI’s embedded devices.

TI's Edge AI solution simplifies the whole product life cycle of DNN development and deployment by providing a rich set of tools and optimized libraries. 

## Overview

The figure below provides a high level summary of the relevant tools:<br><img src="assets/workblocks_tools_software.png" width="800">

## Details of various tools

The table below provides detailed explanation of each of the tools:

| Category                                                | Tool/Link| Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                            | IS NOT                |
|---------------------------------------------------------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| **Model training & associated tools**                   |[tinyml-modelzoo](https://github.com/TexasInstruments/tinyml-modelzoo)| **Model Zoo**<br>- To provide collection of pretrained models and documemtation                                                                                                                                                                                                                                                                                                                                                                    |      |
|ditto                                                         |[Model optimization tools](https://github.com/TexasInstruments/tinyml-modeloptimization)| **Model optimization tools**<br>- **Model surgery**: Modifies models with minimal loss in accuracy and makes it suitable for TI device (replaces unsupported operators)<br>- **Model Pruning/sparsity**: Induces sparsity during training – only applicable for specific devices<br>- **QAT**: **Quantization Aware Training** to improve accuracy with fixed point quantization<br>                                                               |- Does not support Tensorflow   |
|ditto                                                         |[tinyml-torchvision](https://github.com/TexasInstruments/tinyml-torchvision)<br>[tinyml-mmdetection](https://github.com/TexasInstruments/tinyml-mmdetection)<br>[tinyml-yolov5](https://github.com/TexasInstruments/tinyml-yolov5)<br>[tinyml-yolox](https://github.com/TexasInstruments/tinyml-yolox)| Training repositories for various tasks<br>- Provides extensions of popular training repositories (like mmdetection, yolox) with lite version of models                                                                                                                                                                                                                                                                                            |- Does not support Tensorflow   |
| **Inference (and compilation) Tools**                   |[tinyml-tidl-tools](https://github.com/TexasInstruments/tinyml-tidl-tools)| To get familiar with model compilation and inference flow<br>- [Post training quantization](https://github.com/TexasInstruments/tinyml-tidl-tools/blob/master/docs/tidl_fsg_quantization.md)<br>- Benchmark latency with out of box example models (10+)<br>- Compile user / custom model for deployment<br>- Inference of compiled models on X86_PC or TI SOC using file base input and output<br>- Docker for easy development environment setup |- Does not support benchmarking accuracy of models using TIDL with standard datasets, for e.g. - accuracy benchmarking using MS COCO dataset for object detection models. Please refer to tinyml-benchmark for the same.<br>- Does not support Camera, Display and inference based end-to-end pipeline development. Please refer Edge AI SDK for such usage    | 
|ditto                                                         |[tinyml-benchmark](https://github.com/TexasInstruments/tinyml-benchmark)| Bring your own model and compile, benchmark and generate artifacts for deployment on SDK with camera, inference and display (using tinyml-gst-apps)<br>- Comprehends inference pipeline including dataset loading, pre-processing and post-processing<br>- Benchmarking of accuracy and latency with large data sets<br>- Post training quantization<br>- Docker for easy development environment setup                                            |  |
| **Integrated environment for training and compilation** ||                              |
|ditto                                                         |[Edge AI Studio: Model Composer](https://www.ti.com/tool/EDGE-AI-STUDIO)| GUI based Integrated environment for data set capture, annotation, training, compilation with connectivity to TI development board<br>- Bring/Capture your own data, annotate, select a model, perform training and generate artifacts for deployment on SDK<br>- Live preview for quick feedback                                                                                                                                                  |- Does not support Bring Your Own Model workflow  |
|ditto                                                         |[Model Maker](https://github.com/TexasInstruments/tinyml-modelmaker)| Command line Integrated environment for training & compilation<br>- Bring your own data, select a model, perform training and generate artifacts for deployment on SDK<br>- Backend tool for model composer (early availability of features compared to Model Composer )                                                                                                                                                                           |- Does not support Bring Your Own Model workflow  |
|**Edge AI Software Development Kit**| [Devices & SDKs](readme_sdk.md) | SDK to develop end-to-end AI pipeline with camera, inference and display<br>- Different inference runtime: TFLiteRT, ONNXRT, NEO AI DLR, TIDL-RT<br>- Framework: openVX, gstreamer<br>- Device drivers: Camera, display, networking<br>- OS: Linux, RTOS<br>- May other software modeus: codecs, OpenCV,…                                                                                                                                          |   |

## Workflows

Bring your own data (BYOD) workflow:<br><img src="assets/workflow_bring_your_own_data.png" width="600">


<hr>


## Issue Trackers
**Issue tracker for [Edge AI Studio](https://www.ti.com/tool/EDGE-AI-STUDIO)** is listed in its landing page.

**[Issue tracker for TIDL](https://e2e.ti.com/support/processors/f/791/tags/TIDL)**: Please include the tag **TIDL** (as you create a new issue, there is a space to enter tags, at the bottom of the page). 

**[Issue tracker for edge AI SDK](https://e2e.ti.com/support/processors/f/791/tags/tinyml)** Please include the tag **tinyml** (as you create a new issue, there is a space to enter tags, at the bottom of the page). 

**[Issue tracker for ModelZoo, Model Benchmark & Deep Neural Network Training Software](https://e2e.ti.com/support/processors/f/791/tags/MODELZOO):** Please include the tag **MODELZOO** (as you create a new issue, there is a space to enter tags, at the bottom of the page). 

<hr>

## What is New
- [2024-Apr] First release of the toolchain

<hr>


## License
Please see the [LICENSE](./LICENSE) file for more information about the license under which this landing repository is made available. The LICENSE file of each repository mentioned here is inside that repository.
