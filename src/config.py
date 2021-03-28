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
# app configuration

config = {}

# data location
config["DATA_DIR"] = "/app/_data"

# path to pre-trained bert at google storage
# @see https://github.com/google-research/bert
config["GS_BERT_MODEL_PATH"] = "2018_10_18/uncased_L-12_H-768_A-12"

# max length of comments (chars)
config["MAX_TEXT_LENGTH"] = 5120

# max input sequence length (tokens)
config["MAX_SEQ_LENGTH"] = 128
# -


