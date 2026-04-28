import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 确保日志目录存在（默认为当前目录下的 logs 文件夹）
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        os.makedirs(log_dir, exist_ok=True)

        # 使用 RotatingFileHandler 避免单文件过大
        log_file = os.path.join(str(log_dir), f"{name}.log")  # 每个 logger 独立文件
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB 轮转
            backupCount=5,  # 保留 5 个备份
            encoding='utf-8'
        )
        console_handler = logging.StreamHandler()

        # 统一日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

    return logger
