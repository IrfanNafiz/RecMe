from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import TensorBoard

pretrained_model = load_model("model.h5")
tensorboard_cb = TensorBoard(log_dir="model_logs", histogram_freq=1)
