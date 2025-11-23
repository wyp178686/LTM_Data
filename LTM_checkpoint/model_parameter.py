from transformers import BertForMaskedLM, AutoConfig,BertConfig,AutoModelForMaskedLM
import torch

config = BertConfig.from_json_file('/nfs/home/9502_liuyu/wyp/ladder1000/ladder1000_output/checkpoint-109600/config.json')
#/project1/wangyiping/GROVER_fintuning_task/GROVER_pretrain
# 实例化模型
#model = BertForMaskedLM(config=config)
model = AutoModelForMaskedLM.from_pretrained('/nfs/home/9502_liuyu/wyp/ladder1000/ladder1000_output/checkpoint-109600', config=config, local_files_only=True)

# 计算总参数量
# p.numel() 返回张量中元素的总数
# p.requires_grad 检查参数是否是可训练的
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"PyTorch 模型总参数量 (手动计算): {total_params:,} (约 {total_params / 1_000_000:.2f} Million)")

# 你也可以打印每层的参数量
# print("\n每层参数量明细:")
# for name, parameter in model.named_parameters():
#     if parameter.requires_grad:
#         print(f"  {name}: {parameter.numel():,}")