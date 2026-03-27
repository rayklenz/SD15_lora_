import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

class DreamBoothTrainDataset(Dataset):
    def __init__(self, data_path, placeholder_token, class_name, image_size=512):
        self.data_path = Path(data_path)
        self.placeholder_token = placeholder_token
        self.class_name = class_name
        self.image_size = image_size
        self.images = list(self.data_path.glob("*.jpg")) + \
                     list(self.data_path.glob("*.png")) + \
                     list(self.data_path.glob("*.jpeg"))

        print(f"📸 Загружено {len(self.images)} изображений из {self.data_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        img = np.array(img).astype(np.float32) / 127.5 - 1
        img = torch.from_numpy(img).permute(2, 0, 1)

        prompt = f"a photo of a {self.placeholder_token} {self.class_name}"

        return {"pixel_values": img, "prompt": prompt}