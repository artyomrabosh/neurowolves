# coding=utf-8
# Copyright (c) 2020, Sber.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT3"""

import torch

from generate_samples import setup_model, prepare_tokenizer
from pretrain_gpt3 import initialize_distributed
from src.arguments import get_args


def main():
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # get the tokenizer
    tokenizer = prepare_tokenizer(args)
    tokenizer.save_pretrained(args.export_huggingface)

    # Model
    _ = setup_model(args)


if __name__ == "__main__":
    main()
