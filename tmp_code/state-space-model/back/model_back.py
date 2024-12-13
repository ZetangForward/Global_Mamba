from modelzipper.tutils import *

"""
这个注释是为了测试模型手动分卡的结果
"""
# raw_state_dict = AutoModelForCausalLM.from_pretrained("/nvme/hf_models/gpt-neo-1.3B").state_dict()
# model.load_state_dict(raw_state_dict, strict=False)


# # move model to device
# layers_per_device = {local_rank: 24}
# module_names = ['base_model.model.', '']
# module_name = module_names[0] if 0 else module_names[1]

# # 初始化device_map
# device_map = {}

# # 添加特殊的模块到device 1
# device_map[f'{module_name}transformer.wte'] = local_rank + 1
# device_map[f'{module_name}lm_head'] = local_rank + 1
# device_map[f'{module_name}transformer.drop'] = local_rank + 1
# device_map[f'{module_name}transformer.ln_f'] = local_rank + 1

# # 当前的层索引
# current_layer_index = 0

# # 为每个设备生成层的映射
# for device, num_layers in layers_per_device.items():
#     for _ in range(num_layers):
#         device_map[f'{module_name}transformer.h.{current_layer_index}'] = device
#         current_layer_index += 1

# model = dispatch_model(model, device_map=device_map)

# load dataset


"""
Version 1.0 实现版本
这个版本的粒度很粗糙，没有考虑模块优化，
直接使用多个Conv1d实现了之前类似H3的功能。
相比于H3，还缺少了token shift的能力，
只不过进行了简单的channel mixing
"""
class GatedMultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(GatedMultiScaleConv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.ConstantPad1d((kernel_size - 1, 0), 0),
                nn.Conv1d(
                    in_channels, 
                    out_channels // len(kernel_sizes) * 2, 
                    kernel_size, 
                    groups=out_channels // len(kernel_sizes) * 2,
                ),  # Double the output channels
            ) for kernel_size in kernel_sizes
        ])

    def single_tensor_forward(self, x):
        outputs = []
        for conv in self.convs:
            conv_output = conv(x)  # x: [1, 2048, 4096]
            gate, output = torch.split(conv_output, conv_output.size(1) // 2, dim=1)  # Split the output into two equal parts
            gate = torch.sigmoid(gate)
            outputs.append(output * gate)  # [B, L, D // n]
        outputs = torch.cat(outputs, dim=1)  # concate all the dimension hidden states
        return outputs
    
    def multi_tensor_forward(self, x: List[torch.Tensor]):
        outputs = []
        for conv, hidden_states in zip(self.convs, x):
            conv_output = conv[1](hidden_states)
            gate, output = torch.split(conv_output, conv_output.size(1) // 2, dim=1)
            gate = torch.sigmoid(gate)
            outputs.append(output * gate)
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.single_tensor_forward(x)
        return self.multi_tensor_forward(x)