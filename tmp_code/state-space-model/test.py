from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
from models.custom_mamba_v3_fast_dev import CustomMambaForCausalLM
import torch

ckpt="/nvme1/zecheng/ckpt/mamba-370m-hf-longconv-2048-512--from-sk-15b/version_1/checkpoints/last.ckpt"

tokenizer = AutoTokenizer.from_pretrained("/nvme/hf_models/mamba-1.4b-hf")
model = CustomMambaForCausalLM.from_pretrained("/nvme/hf_models/mamba-1.4b-hf").cuda()
input_ids = tokenizer("How are you", return_tensors="pt")["input_ids"].cuda()
out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
