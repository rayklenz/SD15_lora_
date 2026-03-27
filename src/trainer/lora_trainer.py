import torch
from tqdm import tqdm
from diffusers import DDPMScheduler

class LoraTrainer:
    def __init__(self, model, dataloader, config, logger):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.logger = logger

        self.optimizer = torch.optim.AdamW(
            self.model.unet.parameters(),
            lr=config.trainer.learning_rate
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config.model.pretrained_model_name,
            subfolder="scheduler"
        )
        self.device = config.trainer.device

    def train(self):
        global_step = 0
        total_steps = min(len(self.dataloader) * self.config.trainer.num_epochs,
                          self.config.trainer.max_train_steps)

        for epoch in range(self.config.trainer.num_epochs):
            self.model.train()
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(self.device).float()
                prompt = batch["prompt"]

                # Кодируем в латенты через VAE
                with torch.no_grad():
                    latents = self.model.vae.encode(pixel_values).latent_dist.sample() * 0.18215

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                          (latents.shape[0],), device=self.device)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # Текстовые эмбеддинги
                with torch.no_grad():
                    text_input = self.model.pipe.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=self.model.pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
                    text_embeddings = self.model.text_encoder(text_input.input_ids)[0].float()

                noise_pred = self.model(noisy_latents, timesteps, text_embeddings)
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(loss=loss.item())
                self.logger.log_metrics({"loss": loss.item()}, step=global_step)
                global_step += 1

                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

        self.model.unet.save_pretrained(self.config.trainer.save_dir)
        print(f"✅ Модель сохранена в {self.config.trainer.save_dir}")