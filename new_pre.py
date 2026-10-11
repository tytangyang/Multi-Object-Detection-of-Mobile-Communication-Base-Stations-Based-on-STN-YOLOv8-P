import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import YOLO
from thop import profile

def compute_gflops(model, imgsz=640):
    """计算模型的GFLOPs"""
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(next(model.parameters()).device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return flops / 1e9  # 转换为GFLOPs

def prune_model_structured(model, amount=0.2):
    """结构化剪枝 - 同时处理Conv2d和对应的BatchNorm层"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 1. 首先剪枝Conv2d层
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
            mask = module.weight_mask.sum(dim=(1,2,3)) != 0
            
            # 2. 找到对应的BatchNorm层
            parent_name = name.rsplit('.', 1)[0]
            parent = dict(model.named_modules())[parent_name]
            bn_name = name.split('.')[-1].replace('conv', 'bn')
            if hasattr(parent, bn_name):
                bn_layer = getattr(parent, bn_name)
                
                # 3. 创建新的Conv2d层
                new_conv = nn.Conv2d(
                    in_channels=int(mask.sum()),
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None
                )
                new_conv.weight.data = module.weight[mask].data
                if module.bias is not None:
                    new_conv.bias.data = module.bias.data
                
                # 4. 创建新的BatchNorm层
                new_bn = nn.BatchNorm2d(
                    module.out_channels,
                    eps=bn_layer.eps,
                    momentum=bn_layer.momentum,
                    affine=bn_layer.affine,
                    track_running_stats=bn_layer.track_running_stats
                )
                
                # 5. 复制BN层参数
                new_bn.weight.data = bn_layer.weight.data
                new_bn.bias.data = bn_layer.bias.data
                new_bn.running_mean.data = bn_layer.running_mean.data
                new_bn.running_var.data = bn_layer.running_var.data
                
                # 6. 替换原模块
                setattr(parent, name.split('.')[-1], new_conv)
                setattr(parent, bn_name, new_bn)
            else:
                # 如果没有对应的BN层，只替换Conv2d
                new_conv = nn.Conv2d(
                    in_channels=int(mask.sum()),
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None
                )
                new_conv.weight.data = module.weight[mask].data
                if module.bias is not None:
                    new_conv.bias.data = module.bias.data
                setattr(parent, name.split('.')[-1], new_conv)
                
            # 移除剪枝掩码
            prune.remove(module, 'weight')
    return model

# 1. 加载模型
model = YOLO('best.pt').model  # 直接获取PyTorch模型
model.eval()

# 2. 计算原始GFLOPs
original_gflops = compute_gflops(model)
print(f"Original GFLOPs: {original_gflops:.2f}")

# 3. 剪枝模型
pruned_model = prune_model_structured(model, amount=0.3)

# 4. 计算剪枝后GFLOPs
pruned_gflops = compute_gflops(pruned_model)
print(f"Pruned GFLOPs: {pruned_gflops:.2f} (Reduction: {(original_gflops-pruned_gflops)/original_gflops*100:.1f}%)")

# 5. 保存剪枝后模型
torch.save(pruned_model.state_dict(), 'pruned_model.pt')