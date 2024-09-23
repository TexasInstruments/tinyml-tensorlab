
import model_quant_utils
import onnx
import torch
from onnxsim import simplify

#######################################################################################
opset_version = 17


#######################################################################################
# model definition and export (float)
class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3,8,3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU()
        # self.conv2 = torch.nn.Conv2d(8,8,3, bias=False)
        # self.bn2 = torch.nn.BatchNorm2d(8)
        # self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu2(x)
        return x


example_model = ExampleModel()

#######################################################################################
# model training should go in here.
# for this demonstration we are using the untrained model with random parameters.
#######################################################################################


#######################################################################################
# export onnx
example_model.eval()
example_input = torch.rand((1, 3,32,32))
torch.onnx.export(example_model, example_input, 'example_model_float.onnx', opset_version=opset_version)


#######################################################################################
# quantization
# to install this package, from the torchmodelopt folder do,
# pip install -e ./
# in the following repository
# https://bitbucket.itg.ti.com/projects/EDGEAI-ALGO/repos/edgeai-modeloptimization/browse
import edgeai_torchmodelopt

total_epochs = 10

# quantize with ptq
# qconfig_mapping = torch.ao.quantization.get_default_qat_qconfig_mapping()
# # qconfig_settings = torch.ao.quantization.default_per_channel_symmetric_qnnpack_qconfig
# # qconfig_mapping = QConfigMapping().set_global(qconfig_settings)
# prepared_model = prepare_qat_fx(example_model, qconfig_mapping, example_input)


##################################################################################
# quantize with edgeai_torchmodelopt wrapper
# qconfig_type = edgeai_torchmodelopt.xmodelopt.quantization.v2.qconfig.QConfigType.WC8SYMP2_AT8SYMP2
qconfig_type = dict(weight=dict(bitwidth=8, qscheme=torch.per_channel_symmetric, power2_scale=True),
                    activation=dict(bitwidth=8, qscheme=torch.per_tensor_symmetric, power2_scale=True, range_max=None, fixed_range=False))
prepared_model = edgeai_torchmodelopt.xmodelopt.quantization.v2.QATFxModule(example_model, qconfig_type=qconfig_type, total_epochs=total_epochs)


##################################################################################
# here we use Post-Training-Quantization (PTQ), but we can use QAT as well.
for it in range(total_epochs):
    data_input = torch.rand((1, 3,32,32))
    prepared_model(data_input)

torch.onnx.export(prepared_model, example_input, 'example_model_fakeq.onnx', opset_version=opset_version)


# ##################################################################################
# # TODO: remove thse lines - not needed
# # native pytorch int8 quantization
# # convert_custom_config = model_export_torch_utils.get_convert_custom_config()
# # backend_config = torch.ao.quantization.backend_config.get_native_backend_config()
# quantized_model = prepared_model.convert()
# try:
#     torch.onnx.export(quantized_model, example_input, 'example_model_qdq.onnx', opset_version=opset_version)
# except:
#     print('ERROR: converted qdq model could not be exported to onnx')
# ##################################################################################
# # # Convert QDQ format to Int8 format
# import onnxruntime as ort
# so = ort.SessionOptions()
# so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# so.optimized_model_filepath = 'example_model_int8.onnx'
# # logger.info("Inplace conversion of QDQ model to INT8 model at: {}".format(onnx_file))
# ort.InferenceSession('example_model_qdq.onnx', so)
# ##################################################################################


##################################################################################
prepared_model = prepared_model.convert()


##################################################################################
def quantize_replacement_function(model, pattern, *args, remove_qconfig=True, **kwargs):

    model = torch.fx.symbolic_trace(model) if not isinstance(model, torch.fx.GraphModule) else model

    # for qdq model
    # replacement_entries_qdq = [
    #    ([torch.ao.nn.intrinsic.modules.fused.ConvReLU2d,edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize], model_quant_utils.TINPUOffsetScaleShift.from_conv_relu_fq),
    #    ([edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize, torch.nn.BatchNorm2d, edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize], model_quant_utils.TINPUOffsetScaleShift.from_fq_bn_fq),
    #    ([edgeai_torchmodelopt.xmodelopt.quantization.v2.AdaptiveActivationFakeQuantize], model_quant_utils.TINPUOffsetScaleShift.from_fq),
    # }

    # for converted model
    replacement_entries_converted = [
        ([torch.quantize_per_tensor], model_quant_utils.TINPUOffsetScaleShift.from_q),
        ([torch.ao.nn.quantized.modules.batchnorm.BatchNorm2d], model_quant_utils.TINPUOffsetScaleShift.from_qbn),
        ([torch.ao.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d], model_quant_utils.TINPUOffsetScaleShift.from_qconvrelu),
        (['dequantize'], model_quant_utils.TINPUOffsetScaleShift.from_dq)
    ]

    for replacement_pattern, replacement_function in replacement_entries_converted:
        matches = edgeai_torchmodelopt.xmodelopt.surgery.v2.replacer.straight_type_chain_searcher(model, replacement_pattern)
        for no_of_module_replaced, (start, end) in enumerate(matches):
            new_fq_module = replacement_function(model, start, end)
            edgeai_torchmodelopt.xmodelopt.surgery.v2.replacer._replace_pattern(model, start, end, new_fq_module, no_of_module_replaced)
        #
    #

    return model


replacement_dict = {
    'replace_types1': quantize_replacement_function
}
prepared_model.module = edgeai_torchmodelopt.xmodelopt.surgery.v2.convert_to_lite_fx(prepared_model.module, replacement_dict)

##################################################################################
ti_npu_onnx_file = 'example_model_ti_npu.onnx'
torch.onnx.export(prepared_model, example_input, ti_npu_onnx_file, opset_version=opset_version)

# simplify
prepared_model_onnx = onnx.load(ti_npu_onnx_file)
prepared_model_onnx, check = simplify(prepared_model_onnx, skipped_optimizers=['fuse_add_bias_into_conv'])
onnx.save(prepared_model_onnx, ti_npu_onnx_file)
