import hydra
from omegaconf import DictConfig
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import random
import numpy as np


@hydra.main(version_base=None, config_path="src/configs", config_name="persongen_inference_lora_b")
def main(cfg: DictConfig):
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"📥 Загрузка модели {cfg.model.pretrained_model_name}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model.pretrained_model_name,
        torch_dtype=torch.float32,
        safety_checker=None
    )

    lora_path = cfg.inferencer.ckpt_dir
    print(f"📥 Загрузка LoRA из {lora_path}...")
    if os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        print("✅ LoRA загружена!")
    else:
        print(f"⚠️ Файлы LoRA не найдены в {lora_path}")

    pipe.to(cfg.trainer.device)
    pipe.enable_attention_slicing()

    # ========== НАСТРОЙКИ ==========
    GUIDANCE_SCALE = 7.5
    LORA_STRENGTH = 0.75
    NUM_INFERENCE_STEPS = cfg.inferencer.num_inference_steps
    MAX_RETRIES = 3

    # ========== КОРОТКИЕ БУСТЫ ==========
    ANATOMY_BOOST = "detailed face, sharp beak, cere, clear eyes, two legs, zygodactyl feet"
    UNIVERSAL_POSITIVE = "high resolution, 8k, masterpiece, plain background, solid color background"

    # Негативы
    NEGATIVE_BOOST = (
        "blurry, deformed, mutated, low quality, bad anatomy, extra limbs, worst quality, cropped, out of frame, "
        "blue beak, lens flare, bokeh, depth of field, motion blur, light streaks, glitch, "
        "busy background, detailed background, scenery, landscape, "
        "two beaks, extra eyes, eyes on beak, misplaced eyes, beak missing, face deformed, oversized beak, huge beak, "
        "narrow head, narrow forehead, elongated head, flat head, pointed skull, "
        "three legs, four legs, extra leg, wrong number of legs"
    )
    # ========== СТИЛИ БЕЗ КОМПОЗИЦИОННЫХ ВСТАВОК ==========
    styles = [
        {"name": "Нежная лаванда",
         "prompt": f"single mybudgiee budgie, soft lavender purple and light blue pastel feathers, dreamy, kawaii, fluffy, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}",
         "negative": NEGATIVE_BOOST},
        {"name": "Сахарная вата",
         "prompt": f"single mybudgiee budgie, cotton candy pink and mint green, sweet, fluffy texture, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}",
         "negative": NEGATIVE_BOOST},
        {"name": "Лунная нежность",
         "prompt": f"single mybudgiee budgie, ethereal, silver and soft blue glowing feathers, centered, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}",
         "negative": NEGATIVE_BOOST},
        {"name": "Солнечный лучик",
         "prompt": f"single mybudgiee budgie, bright yellow and gold shining feathers, centered, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}",
         "negative": NEGATIVE_BOOST},
        {"name": "Грозовая туча",
         "prompt": f"single mybudgiee budgie, cloud fluffy texture, mysterious, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}",
         "negative": NEGATIVE_BOOST },
        {"name": "Пламя гнева",
         "prompt": f"single mybudgiee budgie, fiery red and orange feathers, centered, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}",
         "negative": NEGATIVE_BOOST + ", deformed, melting, two heads, double head, fused heads, duplicate head, extra head"},
        {"name": "Ночной охотник",
         "prompt": f"single mybudgiee budgie, shadows, misterious, deep purple and black feathers, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}",
         "negative": NEGATIVE_BOOST },
        {"name": "Утренняя роса",
         "prompt": f"single mybudgiee budgie, soft green and blue feathers, water texture, centered, two legs, zygodactyl feet, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}",
         "negative": NEGATIVE_BOOST + ", aggressive, dark, bright, loud"},
        {"name": "Валентинов день",
         "prompt": f"two mybudgiee budgie, romantic scene, red and pink feathers, soft lighting, kissing, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}",
         "negative": NEGATIVE_BOOST + ", aggressive, dark, sad, alone, merged, conjoined"},
        {"name": "Облако влюбленных",
         "prompt": f"two mybudgiee budgie, soft pink cloud texture, romantic, couple, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}",
         "negative": NEGATIVE_BOOST + ", alone, aggressive, dark, merged, conjoined"},
        {"name": "Небесная лазурь",
         "prompt": f"single mybudgiee budgie, sky blue feathers, white face, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}",
         "negative": NEGATIVE_BOOST + ", ugly, blurry"},
        {"name": "Альбинос",
         "prompt": f"single mybudgiee budgie, albino variety, pure white feathers, elegant, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}",
         "negative": NEGATIVE_BOOST + ", ugly, blurry, color"},
        {"name": "Опалин",
         "prompt": f"single mybudgiee budgie, opaline variety, soft gradient colors, pastel, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}",
         "negative": NEGATIVE_BOOST + ", ugly, blurry"},
    ]

    random.shuffle(styles)
    os.makedirs("generated_images_budgie_final_nocomp", exist_ok=True)

    print("\n" + "=" * 60)
    print(f"🎨 ФИНАЛЬНАЯ ГЕНЕРАЦИЯ (без композиционных вставок)")
    print(f"   ✅ LoRA сила: {LORA_STRENGTH}")
    print(f"   ✅ Guidance scale: {GUIDANCE_SCALE}")
    print(f"   ✅ Шагов: {NUM_INFERENCE_STEPS}")
    print(f"   📸 Всего стилей: {len(styles)}")
    print("=" * 60)

    def check_if_cropped(image):
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        edge_top = img_array[0:10, :].mean()
        edge_bottom = img_array[-10:, :].mean()
        edge_left = img_array[:, 0:10].mean()
        edge_right = img_array[:, -10:].mean()
        center = img_array[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20].mean()
        edge_avg = (edge_top + edge_bottom + edge_left + edge_right) / 4
        diff = abs(edge_avg - center)
        top_bright = edge_top > (center * 0.8)
        return (diff < 10) or top_bright

    images = []
    cropped_warning = False

    for i, style in enumerate(styles):
        print(f"\n🎨 {i + 1}/{len(styles)}: {style['name']}")
        print(f"   Промпт: {style['prompt'][:80]}...")

        best_image = None
        best_cropped = True
        for attempt in range(MAX_RETRIES + 1):
            with torch.no_grad():
                image = pipe(
                    style["prompt"],
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    negative_prompt=style["negative"],
                    height=512,
                    width=512,
                    cross_attention_kwargs={"scale": LORA_STRENGTH}
                ).images[0]

            if image.mode != "RGB":
                image = image.convert("RGB")

            if not check_if_cropped(image):
                best_image = image
                best_cropped = False
                break
            else:
                if best_image is None:
                    best_image = image
                print(f"   ⚠️ Попытка {attempt + 1}: обрезано, перегенерируем...")

        if best_cropped:
            cropped_warning = True
            print(f"   ⚠️ После {MAX_RETRIES + 1} попыток изображение может быть обрезано.")
        else:
            print(f"   ✅ Изображение в кадре.")

        images.append(best_image)
        filename = f"generated_images_budgie_final_nocomp/{style['name'].replace(' ', '_')}.png"
        best_image.save(filename)
        print(f"   💾 Сохранено: {filename}")

    print("\n🖼️ СОЗДАЮ КОЛЛАЖ...")
    n = len(styles)
    rows = (n + 5) // 6
    cols = min(6, n)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, (style, img) in enumerate(zip(styles, images)):
        axes[i].imshow(img)
        axes[i].set_title(f"{style['name']}", fontsize=9)
        axes[i].axis('off')

    for i in range(len(images), rows * cols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("generated_images_budgie_final_nocomp/all_styles_collage.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("✅ ГОТОВО! Результаты в папке generated_images_budgie_final_nocomp/")
    print(f"📸 Всего сгенерировано: {len(images)} изображений")


if __name__ == "__main__":
    main()
