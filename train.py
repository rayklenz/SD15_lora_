import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from src.datasets.dreambooth import DreamBoothTrainDataset
from src.model.sd15_lora import SD15LoRA
from src.trainer.lora_trainer import LoraTrainer
from src.logger.console_logger import ConsoleLogger
from src.utils.helpers import set_seed

@hydra.main(version_base=None, config_path="src/configs", config_name="persongen_train_lora")
def main(cfg: DictConfig):
    set_seed(42)

    # Датасет
    dataset = hydra.utils.instantiate(cfg.datasets.train[cfg.train_dataset_name])
    dataloader = DataLoader(dataset, batch_size=cfg.trainer.batch_size, shuffle=True)

    # Модель
    model = hydra.utils.instantiate(cfg.model, device=cfg.trainer.device)

    # Логгер
    logger = ConsoleLogger(cfg)

    # Тренер
    trainer = LoraTrainer(model, dataloader, cfg, logger)
    trainer.train()

if __name__ == "__main__":
    main()