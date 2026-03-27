import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

class SD15LoRA(torch.nn.Module):
    def __init__(self, pretrained_model_name, rank=16, alpha=16, dropout=0.0, device="cpu"):
        super().__init__()
        self.pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.float32,
            safety_checker=None
        )
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae

        # Заморозка всех параметров
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        # Конфигурация LoRA для UNet
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=dropout,
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()

        self.device = device
        self.to(device)

    def forward(self, latents, timesteps, text_embeddings):
        noise_pred = self.unet(
            latents,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        return noise_pred