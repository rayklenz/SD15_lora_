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

@hydra.main(version_base=None, config_path="src/configs", config_name="persongen_inference_lora")
def main(cfg: DictConfig):
    # Фиксируем seed для воспроизводимости
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Загружаем базовую модель
    print(f"📥 Загрузка модели {cfg.model.pretrained_model_name}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model.pretrained_model_name,
        torch_dtype=torch.float32,
        safety_checker=None
    )

    # 2. Загружаем LoRA
    lora_path = cfg.inferencer.ckpt_dir
    print(f"📥 Загрузка LoRA из {lora_path}...")
    if os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        print("✅ LoRA загружена!")
    else:
        print(f"⚠️ Файлы LoRA не найдены в {lora_path}")

    pipe.to(cfg.trainer.device)
    pipe.enable_attention_slicing()

    # =====================================================
    # 🎨 НАСТРОЙКИ
    # =====================================================
    GUIDANCE_SCALE = 7.5
    LORA_STRENGTH = 1.0
    NUM_INFERENCE_STEPS = cfg.inferencer.num_inference_steps
    RETRY_CROPPED = True          # перегенерировать обрезанные кадры
    MAX_RETRIES = 3               # максимум попыток на стиль

    # =====================================================
    # 👁️ ЛИЦО — ОБЯЗАТЕЛЬНО ВЕЗДЕ!
    # =====================================================
    FACE_BOOST = "face visible, bird face, expressive face, detailed face, eyes visible, beak visible, facial features"

    # ПОЗИТИВНЫЕ ПРОМПТЫ ДЛЯ АНАТОМИИ И КОМПОЗИЦИИ
    ANATOMY_BOOST = "perfect bird anatomy, proper proportions, natural pose, realistic structure, two legs, two wings, single beak"
    COMPOSITION_FULL = f"bird fully in frame, entire bird visible, whole body shown, centered, good composition, well-framed, no cropping, head fully visible, head not cropped, full head in frame, {FACE_BOOST}"
    COMPOSITION_PORTRAIT = f"close-up, face visible, head fully in frame, beak visible, portrait composition, centered face, {FACE_BOOST}"
    COMPOSITION_MEDIUM = f"medium shot, upper body visible, well-framed, bird centered, head fully visible, {FACE_BOOST}"
    UNIVERSAL_POSITIVE = "sharp focus, high quality, detailed, beautiful"

    # НЕГАТИВНЫЕ ПРОМПТЫ (усилены)
    ANATOMY_NEGATIVE = "bad anatomy, deformed, distorted, unnatural pose, extra limbs, wrong proportions, malformed, three legs, four legs, extra wings, missing legs, two beaks, extra beak, no beak"
    COMPOSITION_NEGATIVE = "cropped, out of frame, partially visible, incomplete, truncated, cut off, missing parts, head cut off, body cut off, edge of frame, frame edge, head out of frame, beak cut off"
    FACE_NEGATIVE = "no face, face hidden, face obscured, head hidden, no eyes, face not visible"
    QUALITY_NEGATIVE = "blurry, low quality, ugly, distorted, pixelated, artifacts"
    STYLE_NEGATIVE = "oversaturated, neon, rainbow, gaudy, spiky, thorny, cactus-like, rough texture, heavy, solid"

    NEGATIVE_BOOST = f"{ANATOMY_NEGATIVE}, {COMPOSITION_NEGATIVE}, {FACE_NEGATIVE}, {QUALITY_NEGATIVE}, {STYLE_NEGATIVE}"

    # =====================================================
    # 🎭 СТИЛИ С УЛУЧШЕННЫМИ ПРОМПТАМИ
    # =====================================================
    styles = [
        # ========== ПОЛНЫЙ КАДР ==========
        {
            "name": "Реалистичный (полный)",
            "prompt": f"a photo of a sks bird, full body, entire bird visible, face visible, detailed feathers, professional photography, 4k, realistic, natural lighting, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", cartoon, painting, sketch, close-up, portrait"
        },
        {
            "name": "Художественный (полный)",
            "prompt": f"a beautiful sks bird, full body, entire bird visible, face visible, watercolor painting style, artistic, soft colors, elegant, full head, not cropped, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", photo, realistic, close-up, portrait"
        },
        {
            "name": "Мультяшный (полный)",
            "prompt": f"a cute sks bird, full body, entire bird visible, face visible, cartoon style, pixar style, colorful, adorable, animation, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", realistic, photo, close-up, intricate pattern, textured background"
        },
        {
            "name": "Фэнтези (полный)",
            "prompt": f"a magical sks bird, fantasy art, ethereal, glowing feathers, mystical, shimmering, iridescent, soft lighting, delicate, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", realistic, ugly, oversaturated, neon, spiky, heavy"
        },  #!!!!!!!!!!!!!!!!
        {
            "name": "Акварель (полный)",
            "prompt": f"a sks bird, full body, entire bird visible, face visible, watercolor painting, wet on wet, artistic, soft textures, beautiful, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", photo, realistic, close-up"
        },
        {
            "name": "Винтажный (полный)",
            "prompt": f"a sks bird, full body, entire bird visible, face visible, vintage photo, sepia tones, old photograph style, nostalgic, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", cartoon, close-up, portrait"
        },

        # ========== ПОРТРЕТ ==========
        {
            "name": "Реалистичный (портрет)",
            "prompt": f"a photo of a sks bird, close-up portrait, face visible, detailed feathers around face, professional photography, 4k, realistic, natural lighting, beak slightly below eye level, {ANATOMY_BOOST}, {COMPOSITION_PORTRAIT}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", cartoon, painting, sketch, full body, far away, cropped, cut off"
        },
        {
            "name": "Художественный (портрет)",
            "prompt": f"a beautiful sks bird, close-up portrait, face visible, watercolor painting style, artistic, soft colors, elegant, full head, beak visible, {ANATOMY_BOOST}, {COMPOSITION_PORTRAIT}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", photo, realistic, full body, far away, cropped"
        },
        {
            "name": "Мультяшный (портрет)",
            "prompt": f"a cute sks bird, close-up portrait, face visible, cartoon style, pixar style, colorful, adorable, animation, detailed eyes, expressive, {ANATOMY_BOOST}, {COMPOSITION_PORTRAIT}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", realistic, photo, full body, far away, oversized eyes, giant eyes"
        },
        {
            "name": "Фэнтези (портрет)",
            "prompt": f"a magical sks bird, close-up portrait, face visible, fantasy art, glowing feathers, mystical, ethereal, shimmering, iridescent, soft lighting, delicate, {ANATOMY_BOOST}, {COMPOSITION_PORTRAIT}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", realistic, full body, far away, oversaturated, neon, rainbow"
        },
        {
            "name": "Акварель (портрет)",
            "prompt": f"a sks bird, close-up portrait, face visible, watercolor painting, wet on wet, artistic, soft textures, beautiful, {ANATOMY_BOOST}, {COMPOSITION_PORTRAIT}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", photo, realistic, full body"
        },
        {
            "name": "Винтажный (портрет)",
            "prompt": f"a sks bird, close-up portrait, face visible, vintage photo, sepia tones, old photograph style, nostalgic, full head, beak inside frame, {ANATOMY_BOOST}, {COMPOSITION_PORTRAIT}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", cartoon, full body, far away"
        },

        # ========== СРЕДНИЙ ПЛАН ==========
        {
            "name": "Реалистичный (средний)",
            "prompt": f"a photo of a sks bird, medium shot, upper body visible, face visible, detailed feathers, professional photography, 4k, realistic, single beak, {ANATOMY_BOOST}, {COMPOSITION_MEDIUM}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", cartoon, painting, sketch, close-up, far away, two beaks"
        },
        {
            "name": "Художественный (средний)",
            "prompt": f"a beautiful sks bird, medium shot, upper body visible, face visible, watercolor painting style, artistic, soft colors, {ANATOMY_BOOST}, {COMPOSITION_MEDIUM}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", photo, realistic, close-up, far away"
        },
        {
            "name": "Мультяшный (средний)",
            "prompt": f"a cute sks bird, medium shot, upper body visible, face visible, cartoon style, pixar style, colorful, adorable, {ANATOMY_BOOST}, {COMPOSITION_MEDIUM}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", realistic, photo, close-up, far away"
        },
        {
            "name": "Фэнтези (средний)",
            "prompt": f"a magical sks bird, fantasy art, ethereal, glowing feathers, mystical, shimmering, iridescent, soft lighting, delicate, {ANATOMY_BOOST}, {COMPOSITION_MEDIUM}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", realistic, ugly, oversaturated, neon, spiky, heavy"
        }, #!!!!!
        {
            "name": "Акварель (средний)",
            "prompt": f"a sks bird, medium shot, upper body visible, face visible, watercolor painting, wet on wet, artistic, soft textures, {ANATOMY_BOOST}, {COMPOSITION_MEDIUM}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", photo, realistic, close-up"
        },
        {
            "name": "Винтажный (средний)",
            "prompt": f"a sks bird, medium shot, upper body visible, face visible, vintage photo, sepia tones, old photograph style, {ANATOMY_BOOST}, {COMPOSITION_MEDIUM}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", cartoon, close-up, far away"
        }
    ]

    # Перемешивание стилей (как и было)
    random.shuffle(styles)
    print("🎲 Стили перемешаны в случайном порядке!")

    # Фильтры (оставлены закомментированными)
    # styles = [s for s in styles if "полный" in s["name"]]
    # styles = [s for s in styles if "портрет" in s["name"]]
    # styles = [s for s in styles if "средний" in s["name"]]
    # styles = styles[:6]

    os.makedirs("generated_images", exist_ok=True)

    print("\n" + "=" * 60)
    print(f"🎨 МУЛЬТИ-СТИЛЕВАЯ ГЕНЕРАЦИЯ (улучшенная)")
    print(f"   ✅ LoRA сила: {LORA_STRENGTH}")
    print(f"   ✅ Guidance scale: {GUIDANCE_SCALE}")
    print(f"   👁️ ЛИЦО: ОБЯЗАТЕЛЬНО во всех кадрах!")
    print(f"   🛡️ Защита от обрезания: ВКЛЮЧЕНА (автоперегенерация)")
    print(f"   🎯 Типы кадров: полный | портрет | средний")
    print(f"   🎲 Порядок: СЛУЧАЙНЫЙ")
    print(f"   📸 Всего стилей: {len(styles)}")
    print(f"   🔢 Seed: {seed}")
    print("=" * 60)

    # =====================================================
    # 🔍 ФУНКЦИЯ ПРОВЕРКИ ОБРЕЗАНИЯ
    # =====================================================
    def check_if_cropped(image, threshold=0.05):
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        # Проверяем края
        edge_top = img_array[0:10, :].mean()
        edge_bottom = img_array[-10:, :].mean()
        edge_left = img_array[:, 0:10].mean()
        edge_right = img_array[:, -10:].mean()
        center = img_array[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20].mean()
        edge_avg = (edge_top + edge_bottom + edge_left + edge_right) / 4
        diff = abs(edge_avg - center)
        # Дополнительно: если верхний край яркий (значит там не фон), то считаем обрезанным
        top_bright = edge_top > (center * 0.8)  # порог подберите
        return (diff < 10) or top_bright

    # =====================================================
    # 🎨 ГЕНЕРАЦИЯ С ПОВТОРАМИ ПРИ ОБРЕЗАНИИ
    # =====================================================
    images = []
    cropped_warning = False

    for i, style in enumerate(styles):
        print(f"\n🎨 {i+1}/{len(styles)}: {style['name']}")
        print(f"   Промпт: {style['prompt'][:80]}...")

        best_image = None
        best_cropped = True
        for attempt in range(MAX_RETRIES+1):
            # Генерируем изображение
            with torch.no_grad():
                image = pipe(
                    style["prompt"],
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    negative_prompt=style["negative"],
                    height=768,
                    width=768,
                    cross_attention_kwargs={"scale": LORA_STRENGTH}
                ).images[0]

            if image.mode != "RGB":
                image = image.convert("RGB")

            is_cropped = check_if_cropped(image)
            if not is_cropped:
                best_image = image
                best_cropped = False
                break
            else:
                # Если обрезано, запоминаем первое на случай, если все попытки неудачны
                if best_image is None:
                    best_image = image
                print(f"   ⚠️ Попытка {attempt+1}: обрезано, перегенерируем...")

        if best_cropped:
            cropped_warning = True
            print(f"   ⚠️ После {MAX_RETRIES+1} попыток изображение всё ещё может быть обрезано.")
        else:
            print(f"   ✅ Изображение в кадре.")

        images.append(best_image)
        filename = f"generated_images/{style['name'].replace(' ', '_')}.png"
        best_image.save(filename)
        print(f"   💾 Сохранено: {filename}")

    # =====================================================
    # 🖼️ ВИЗУАЛИЗАЦИЯ
    # =====================================================
    print("\n🖼️ СОЗДАЮ КОЛЛАЖ...")
    n = len(styles)
    rows = (n + 5) // 6
    cols = min(6, n)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3*rows))
    if rows*cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (style, img) in enumerate(zip(styles, images)):
        if "портрет" in style["name"]:
            color = 'darkred'
            frame_type = "🔍 Портрет"
        elif "полный" in style["name"]:
            color = 'darkgreen'
            frame_type = "🦜 Полный"
        else:
            color = 'darkblue'
            frame_type = "📷 Средний"
        axes[i].imshow(img)
        axes[i].set_title(f"{frame_type}: {style['name']}", fontsize=9, color=color)
        axes[i].axis('off')

    for i in range(len(images), rows*cols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("generated_images/all_styles_collage.png", dpi=150, bbox_inches='tight')
    plt.show()

    # =====================================================
    # 📊 СТАТИСТИКА
    # =====================================================
    print("\n" + "=" * 60)
    print("✅ ГОТОВО!")
    print("=" * 60)
    print(f"\n📁 Результаты сохранены в папке: generated_images/")
    print(f"📸 Всего сгенерировано: {len(images)} изображений")
    if cropped_warning:
        print("\n⚠️ Некоторые изображения могут быть обрезаны (несмотря на попытки перегенерации).")
    else:
        print("\n✅ Все изображения выглядят корректно!")

    full_count = len([s for s in styles if "полный" in s["name"]])
    portrait_count = len([s for s in styles if "портрет" in s["name"]])
    medium_count = len([s for s in styles if "средний" in s["name"]])
    print(f"\n🎞️ Типы кадров в коллекции:")
    print(f"   🦜 Полный кадр (с лицом): {full_count}")
    print(f"   🔍 Портрет (лицо крупно): {portrait_count}")
    print(f"   📷 Средний план (с лицом): {medium_count}")

if __name__ == "__main__":
    main()