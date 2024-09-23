import torch


def compute_shift_scale(x, num_bits_shift=8, num_bits_scale=8, print_mse =True):
    x_abs = x.abs()
    x_sign = x.sign()
    u_p = 2**(num_bits_scale)-1
    power_of_2 = torch.floor(torch.log2(u_p/x_abs))*torch.tensor([1.0])
    #print('power_of_2',power_of_2)
    shift = power_of_2.clamp(min=0, max=(2**(num_bits_shift)-1))
    #print('shift',shift)
    scale = x_abs * torch.pow(torch.tensor([2.0]), shift)
    #print('scale',scale)

    #nan fix:
    mask = torch.isnan(scale)
    #print(mask)
    scale[mask] = 0
    shift[mask] = 1

    assert(torch.sum(scale > u_p) == 0)

    scale = x_sign*torch.round(scale)

    # x_hat = scale *  torch.pow(torch.tensor([2.0]), -shift)
    # mse = torch.mean((x-x_hat)**2)
    # if print_mse:
    #     print(mse)
    return shift, scale


class TINPUOffsetScaleShift(torch.nn.Module):
    def __init__(self, offset, scale, shift, c_n, c_p, quantize_per_channel=False, use_floor=True):
        super().__init__()
        self.c_n = c_n
        self.c_p = c_p
        self.quantize_per_channel = quantize_per_channel
        self.use_floor = use_floor
        self.register_buffer('offset', offset)
        self.register_buffer('scale', scale)
        self.register_buffer('shift', shift)

    def extra_repr(self):
        return 'num_bits_offset={num_bits_offset}, num_bits_scale={num_bits_scale}, num_bits_shift={num_bits_shift}, clip={c_p}'.format(**self.__dict__)

    def forward(self, x):
        y = (x + self.offset.reshape(1,-1,1,1))*self.scale.reshape(1,-1,1,1)
        y = y * self.shift.reshape(1,-1,1,1)
        if self.use_floor:
            y = torch.floor(y).clamp(min=self.c_n, max=self.c_p) #the floor operation mimics the actual shift and bit select in hardware
        else:
            y = torch.round(y).clamp(min=self.c_n,max=self.c_p)
        return y

    @staticmethod
    def from_q(model, start, end):
        q_node = start
        scale = getattr(model, q_node.args[1].target)
        zero_point = getattr(model, q_node.args[2].target)
        id_module = torch.nn.Identity()
        id_module.scale = scale
        id_module.zero_point = zero_point
        return id_module

    @staticmethod
    def from_qbn(model, start, end):
        zero_point_offset_for_activation = -128

        qbn_module = dict(model.named_modules())[start.target]
        bn_sigma = torch.sqrt(qbn_module.running_var + qbn_module.eps)
        bn_weight = (qbn_module.weight / bn_sigma)
        bn_bias = qbn_module.bias - (qbn_module.running_mean / bn_sigma)
        scale2 = qbn_module.scale
        zero_point2 = qbn_module.zero_point

        combined_weight = bn_weight / scale2
        oss_shift, oss_scale = compute_shift_scale(combined_weight)
        oss_offset = bn_bias / scale2 + zero_point2 + zero_point_offset_for_activation

        oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -128, 127)
        oss_module.scale = qbn_module.scale
        oss_module.zero_point = qbn_module.zero_point
        return oss_module

    @staticmethod
    def from_qconvrelu(model, start, end):
        qconvrelu_module = dict(model.named_modules())[start.target]
        conv_module = torch.nn.Conv2d(qconvrelu_module.in_channels, qconvrelu_module.out_channels, qconvrelu_module.kernel_size, bias=False)

        weight = qconvrelu_module.weight()
        per_channel = (weight.qscheme() in (torch.per_channel_symmetric, torch.per_channel_affine))

        qweight = weight.data.detach().int_repr()
        conv_module.weight.data.copy_(qweight)

        weight_scale = weight.q_per_channel_scales() if per_channel else weight.q_scale()
        weight_zero_point = weight.q_per_channel_zero_points() if per_channel else weight.q_zero_point()
        input_scale = dict(model.named_modules())[start.prev.target].scale
        input_zero_point = dict(model.named_modules())[start.prev.target].zero_point

        bias_scale = weight_scale * input_scale
        bias_zero_point = weight_zero_point
        bias = qconvrelu_module.bias()

        # qbias = (torch.round(bias / bias_scale) + bias_zero_point).float()
        if per_channel:
            qbias = torch.quantize_per_channel(bias, bias_scale, bias_zero_point, 0, torch.qint32)
        else:
            qbias = torch.quantize_per_tensor(bias, bias_scale, bias_zero_point, 0, torch.qint32)
        #
        qbias = qbias.int_repr()

        # conv_module.bias.data.copy_(qbias)
        oss_shift, oss_scale = compute_shift_scale(torch.tensor(qconvrelu_module.scale))
        oss_module = TINPUOffsetScaleShift(qbias, oss_scale, oss_shift, -255, 255)

        seq_module = torch.nn.Sequential(conv_module, oss_module, torch.nn.ReLU(), torch.nn.Hardtanh(0, 255))
        seq_module.scale = qconvrelu_module.scale
        seq_module.zero_point = qconvrelu_module.zero_point
        return seq_module

    @staticmethod
    def from_dq(model, start, end):
        id_module = torch.nn.Identity()
        id_module.scale = 1.0
        id_module.zero_point = 0.0
        return id_module

    # @staticmethod
    # def from_fq(model, start, end):
    #     fq_module1 = dict(model.named_modules())[start.target]
    #     fq_observed1 = fq_module1.activation_post_process
    #     scale, zero_point = fq_observed1.calculate_qparams()
    #     device = scale.device
    #     oss_offset = torch.tensor(0.0,device=device)
    #     oss_scale = torch.tensor(1.0,device=device)
    #     oss_shift = torch.tensor(1.0,device=device)
    #     oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, 0, 255)
    #     return oss_module

    # @staticmethod
    # def from_fq_bn_fq(model, start, end):
    #     fq_module1 = dict(model.named_modules())[start.target]
    #     bn_module = dict(model.named_modules())[start.next.target]
    #     fq_module2 = dict(model.named_modules())[end.target]
    #
    #     zero_point_offset_for_activation = -128
    #     fq_observed1 = fq_module1.activation_post_process
    #     scale1, zero_point1 = fq_observed1.calculate_qparams()
    #
    #     bn_sigma = torch.sqrt(bn_module.running_var + bn_module.eps)
    #     bn_weight = (bn_module.weight / bn_sigma)
    #     bn_bias = bn_module.bias - (bn_module.running_mean / bn_sigma)
    #
    #     fq_observed2 = fq_module2.activation_post_process
    #     scale2, zero_point2 = fq_observed2.calculate_qparams()
    #     device = scale2.device
    #
    #     combined_weight = bn_weight / scale2
    #     oss_shift, oss_scale = compute_shift_scale(combined_weight)
    #     oss_offset = bn_bias / scale2 + zero_point2 + zero_point_offset_for_activation
    #
    #     oss_module = TINPUOffsetScaleShift(oss_offset, oss_scale, oss_shift, -128, 127)
    #     return oss_module

    # @staticmethod
    # def from_conv_relu_fq(model, start, end):
    #     conv_module = conv_relu_module[0]
    #     weight_scale = conv_module.weight_scale
    #     weight_zero_point = conv_module.weight_zero_point
    #     device = weight_scale.device
    #     conv_int_in_float = torch.nn.Conv2d(conv_module.in_channels, conv_module.out_channels, conv_module.kernel_size)
    #     oss_module = __class__.from_fq(fq_module)
    #     relu_module = torch.nn.ReLU()
    #     return torch.nn.Sequential(conv_int_in_float, oss_module, relu_module)

    # @staticmethod
    # def from_cbn_fq(model, start, end):
    #     qconv_module = dict(model.named_modules())[start.target]
    #     fq_module = dict(model.named_modules())[end.target]
    #     fq_observed = fq_module.activation_post_process
    #     scale, zero_point = fq_observed.calculate_qparams()
    #     device = scale.device
    #     oss_offset = torch.tensor(0.0,device=device)
    #     oss_scale = torch.tensor(0.0,device=device)
    #     oss_shift = torch.tensor(0.0,device=device)
    #     conv_module = cbn_module.to_float()
    #     conv_module = torch.nn.Conv2d()
    #     return conv_module
