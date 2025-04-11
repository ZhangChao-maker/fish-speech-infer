import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # 全局关闭
import numpy as np
import soundfile as sf
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import hydra

# 初始化配置
hydra.core.global_hydra.GlobalHydra.instance().clear()
with hydra.initialize(version_base="1.3", config_path="./configs"):
    cfg = hydra.compose(config_name="firefly_gan_vq")

# 硬编码参数
checkpoint_path = "../checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
input_npy = "../temp/codes_1.npy"
output_wav = "output_audio.wav"
device = "cuda"

# 加载模型
model = instantiate(cfg)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict.get("state_dict", state_dict), strict=False)
model.eval().to(device)

# 加载预计算索引
indices = torch.from_numpy(np.load(input_npy)).to(device).long()

# 生成音频
with torch.no_grad():
    fake_audios, _ = model.decode(
        indices=indices[None], 
        feature_lengths=torch.tensor([indices.shape[1]], device=device)
    )

# 保存结果
sf.write(output_wav, 
        fake_audios[0, 0].float().cpu().numpy(), 
        model.spec_transform.sample_rate)