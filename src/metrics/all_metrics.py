import torch
from transformers import CLIPProcessor, CLIPModel

class MetricsCalculator:
    def __init__(self, device="cpu"):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def compute_textual_similarity(self, images, prompts):
        """Возвращает среднее сходство между изображениями и текстовыми промптами"""
        inputs = self.clip_processor(text=prompts, images=images, return_tensors="pt", padding=True).to(self.device)
        outputs = self.clip_model(**inputs)
        return outputs.logits_per_image.mean().item()

    def compute_image_similarity(self, images1, images2):
        """Сходство между двумя наборами изображений (например, сгенерированными и референсными)"""
        inputs1 = self.clip_processor(images=images1, return_tensors="pt").to(self.device)
        inputs2 = self.clip_processor(images=images2, return_tensors="pt").to(self.device)
        img_embeds1 = self.clip_model.get_image_features(**inputs1)
        img_embeds2 = self.clip_model.get_image_features(**inputs2)
        similarity = torch.cosine_similarity(img_embeds1, img_embeds2).mean().item()
        return similarity