# ==== перенаправлення TF/absl/warnings у лог-файл ====

import os
import warnings
import logging
import absl.logging

# файл логів
LOG_FILE = "trends.log"

# очистити лог перед стартом
open(LOG_FILE, "w").close()

# Налаштування логування у файл
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode="a",
    encoding="utf-8",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# робимо так, щоб TensorFlow писав у logging, а не в stdout
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"           # показувати всі TF-попередження
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"          # прибрати oneDNN noise

# перенаправити absl (TF/Keras) у logging
absl.logging.get_absl_handler().use_absl_log_file(LOG_FILE)
absl.logging.set_verbosity(absl.logging.INFO)

# перенаправити warnings у logging
def warn_to_log(message, category, filename, lineno, file=None, line=None):
    logging.warning(f"{category.__name__}: {message} ({filename}:{lineno})")

warnings.showwarning = warn_to_log

# ==== тільки після цього імпортуємо CLI ====

from trendsai.cli import run

if __name__ == "__main__":
    run()
