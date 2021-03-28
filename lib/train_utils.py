# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import shutil
from datetime import datetime
from pprint import pformat
import argparse
from src.config import config


# tensorboard log dir
def create_tensorboard_run_dir(run):
    tb_log_dir = f"/app/.tensorboard/{run}"
    shutil.rmtree(tb_log_dir, ignore_errors=True)
    return tb_log_dir


# save trained model
def save_trained_model(model, run, info):

    # model
    output_dir = config["DATA_DIR"] + "/saved_models"
    model_path = output_dir + f"/model.{run}.tf"
    os.makedirs(output_dir, exist_ok=True)
    model.save(model_path, overwrite=True, save_format="tf", include_optimizer=True)
    print(f"* Model saved to {model_path}")

    # info
    info_path = model_path + "_info.txt"
    print(
        f"Date:\n\n{datetime.now()} UTC\n\nArguments:\n\n{pformat(info)}",
        file=open(info_path, "w"),
    )


# get_training_arguments
def get_training_arguments(description="resnet50", RUN="A", LR_START=5e-6, VAL_SPLIT=0.2, BATCH_SIZE=64, TOTAL_EPOCHS=50, EARLY_STOP_PATIENCE=10):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f", "--run", type=str, default=RUN)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr_start", type=float, default=LR_START)
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
    parser.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE)

    args = parser.parse_args()
    return args
# -




