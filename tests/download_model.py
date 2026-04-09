# 模型下载
from modelscope import snapshot_download

model_dir = snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch')
print(model_dir)
