# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

MODEL_PATH='/home/daniel/KoSpeech/model.pt'
AUDIO_PATH='/home/daniel/KoSpeech/hi2.pcm'
DEVICE='cpu'

# shellcheck disable=SC2164
cd bin
python run_pretrain.py --model_path $MODEL_PATH --audio_path $AUDIO_PATH --device $DEVICE

# shellcheck disable=SC2103
cd ..
