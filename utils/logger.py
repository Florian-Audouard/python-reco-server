import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("report.log", mode="w"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)
