"""
Run inference on a FMNIST ONNX model specified as the argument
Contributed by Ajay Jayaraj (ajayj@ti.com)
"""
import sys
import timeit
import argparse
import numpy as np
import onnxruntime as ort


def get_args():
    parser = argparse.ArgumentParser(description="Run inference with FMNIST ONNX model")
    parser.add_argument("--model-name", type=str, required=True, help="ONNX model name")

    args = parser.parse_args()

    return args


model_name = get_args().model_name

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

INPUT_NAME = "input"


# Use as decorator - @timethis
def timethis(func):
    def timed(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        print('Function', func.__name__, 'time:', round((end - start) * 1000, 4), 'ms')
        return result

    return timed


#
# Load the image (Ankle boot)
#
flattened_input = np.fromfile('input.dat', dtype=np.float32)
input_image = flattened_input.reshape((1, 28, 28))

#
# Run inference using ONNX RT
#

# Set graph optimization level
ort_session_options = ort.SessionOptions()
ort_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

ort_sess = ort.InferenceSession(model_name, ort_session_options)


@timethis
def run_onnxrt(ort_session):
    prediction = ort_session.run(None, {INPUT_NAME: input_image})
    return prediction


outputs = run_onnxrt(ort_sess)

# Outputs - 1x10
conf_1 = outputs[0][0]
print(conf_1)
print(classes[conf_1.argmax(0)])

sys.exit(0)
