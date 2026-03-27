class ConsoleLogger:
    def __init__(self, config):
        self.config = config

    def log_metrics(self, metrics, step=None):
        print(f"[Step {step}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    def log_images(self, images, caption, step=None):
        import os
        os.makedirs("logs", exist_ok=True)
        for i, img in enumerate(images):
            img.save(f"logs/{caption}_{step}_{i}.png")