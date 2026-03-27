import torch
import json
import os
from pathlib import Path

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
            
class BaseInferencer(BaseTrainer):
    def __init__(
        self,
        model,
        pipe,
        metrics,
        global_config,
        device,
        dataloaders,
        logger,
        writer,
        batch_transforms,
        # inferencer args
        epoch_len,
        epochs_to_infer,
        ckpt_dir,
        exp_save_dir,
        seed,
    ):  
        self.is_train = True

        self.config = global_config
        self.device = device

        self.logger = logger

        self.model = model
        self.pipe = pipe
        self.batch_transforms = batch_transforms
        self.writer = writer
        self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define metrics
        self.metrics = metrics
        self.evaluation_metrics = MetricTracker()
        
        self.epochs_to_infer = epochs_to_infer
        self.ckpt_dir = ckpt_dir
        self.exp_save_dir = exp_save_dir
        self.seed = seed       
    
    def inference(self):
        """
        Full inference logic

        """
        for epoch in self.epochs_to_infer:
            self._last_epoch = epoch
            result = self._inference_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")
                    
    
    def _inference_epoch(self, epoch):
        """
        Inference logic for an particular echeckpoint.

        Args:
            epoch (int): number of current checkpoint.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        logs = {}
        self.is_train = False

        self.writer.set_step(epoch  * self.epoch_len)
        self.writer.add_scalar("general/epoch", epoch)
            
        if epoch != 0:
            ckpt_pth = Path(self.ckpt_dir) / f"checkpoint-epoch{epoch}.pth"
            self._from_pretrained(ckpt_pth)
        
        for part, dataloader in self.evaluation_dataloaders.items():
            self.images_storage = []
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            self.save_results(epoch, part)
            logs.update(**{f"{part}/{name}": value for name, value in val_logs.items()})

        return logs
    
    def store_batch(self, images, prompt):
        """
        Stora batch of images and prompts to process them lates

        Args:
            images (list[PIL.Image]): batch of generated images.
            prompt (str): prompt of generated images
        """
        self.images_storage.append((prompt, images))

    def save_results(self, epoch, part):
        """
        Process gathered data (save images, metrics and etc.)

        Args:
            epoch (int): number of current checkpoint.
            part (str): partition to evaluate on
        """
        
        output_dir = Path(self.exp_save_dir) / f"checkpoint_{epoch}/{part}"
        os.makedirs(output_dir, exist_ok=True)
        metrics_dict = {
            "prompts": []
        }
        
        for prompt, images_batch in self.images_storage:
            metrics_dict["prompts"].append(prompt)
            batch_dir = output_dir/ prompt.replace(" ", "_")
            os.makedirs(batch_dir, exist_ok=True)
            
            for i, image in enumerate(images_batch):
                imgae_pth = batch_dir / f"{i}.jpg"
                image.save(imgae_pth)
        metrics_dict.update(self.evaluation_metrics._data)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics_dict, f)


class LoraInferencer(BaseInferencer):
    def init(self, pipeline, metrics, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=1, **kwargs):
        super().init(**kwargs)
        self.pipeline = pipeline
        self.metrics = metrics
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.num_images_per_prompt = num_images_per_prompt

    @torch.no_grad()
    def process_evaluation_batch(self, batch, eval_metrics):
        # Извлекаем промпты из batch (ожидается список строк)
        prompts = batch["prompt"]

        # Генерируем изображения через pipeline
        output = self.pipeline(
            prompt=prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            num_images_per_prompt=self.num_images_per_prompt
        )

        # Извлекаем изображения из вывода pipeline (обычно это атрибут .images)
        if hasattr(output, 'images'):
            generated_images = output.images
        else:
            # На случай, если pipeline возвращает тензоры или список напрямую
            generated_images = output

        # Сохраняем сгенерированные изображения в batch
        batch['generated'] = generated_images

        # Вычисляем метрики
        for metric in self.metrics:
            # Метрики ожидают, что в batch есть ключи 'image' (оригинал) и 'generated' (сгенерированное)
            # Предполагается, что метрика принимает именованные аргументы, соответствующие ключам batch.
            # Если метрика ожидает batch целиком, замените metric(**batch) на metric(batch).
            metric_result = metric(**batch)
            for k, v in metric_result.items():
                eval_metrics.update(k, v)

        # Сохраняем batch (например, для логирования изображений)
        self.store_batch(generated_images, prompts)

        return batch