# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import yaml
from transformers.trainer_utils import get_last_checkpoint
import random
import numpy as np
import torch


class ProcessorWrapper:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, tensor):
        return self.processor(tensor, return_tensors="pt")["pixel_values"].squeeze(0)


# --- [ 수정 시작: 함수 시그니처 및 로직 변경 ] ---
def possible_override_args(override_args, model_args, data_args, training_args):
    """
    YAML 설정 파일이 있으면 해당 내용으로 model_args, data_args, training_args 객체의 속성을 덮어씁니다.
    """
    if hasattr(override_args, "config_file") and override_args.config_file is not None:
        # 설정 파일 경로 확인 (상대 경로일 경우 configs/ 폴더 기준)
        yaml_file = override_args.config_file
        if not os.path.isabs(yaml_file) and not yaml_file.startswith("configs/"):
             yaml_file = os.path.join("configs", yaml_file)

        # 설정 파일 로드
        try:
            with open(yaml_file, "r") as file:
                config = yaml.safe_load(file)

            if config: # 설정 파일 내용이 비어있지 않은 경우
                 print(f"Overriding arguments with config file: {yaml_file}")
                 # 모든 인자 객체 리스트
                 arg_objects = [model_args, data_args, training_args]

                 # YAML 파일의 각 키-값 쌍에 대해
                 for key, value in config.items():
                      overridden = False
                      # 각 인자 객체를 순회하며 해당 키가 속성으로 존재하는지 확인
                      for arg_obj in arg_objects:
                           if hasattr(arg_obj, key):
                                current_value = getattr(arg_obj, key)
                                # 타입이 다르거나 값이 다를 경우 덮어쓰기 (로깅 추가)
                                if type(current_value) != type(value) or current_value != value:
                                     print(f"  - Overriding '{key}': {current_value} -> {value}")
                                setattr(arg_obj, key, value)
                                overridden = True
                                break # 해당 키를 찾으면 다음 키로 넘어감
                      # if not overridden:
                      #      print(f"  - Warning: Key '{key}' from config not found in any argument object.")
            else:
                 print(f"Warning: Config file '{yaml_file}' is empty.")

        except FileNotFoundError:
            print(f"Warning: Config file not found at '{yaml_file}'. Using default/command-line args.")
        except yaml.YAMLError as exc:
            print(f"Error loading config file '{yaml_file}': {exc}")
        except Exception as e:
            print(f"An unexpected error occurred while processing config file '{yaml_file}': {e}")


    # 수정된 인자 객체들을 튜플로 반환
    return model_args, data_args, training_args
# --- [ 수정 종료 ] ---


def get_full_dirs(training_args):
    if not os.path.isabs(training_args.output_dir):
        training_args.output_dir = os.path.join(
            training_args.base_dir, training_args.output_dir
        )
    if not os.path.isabs(training_args.data_dir):
        training_args.data_dir = os.path.join(
            training_args.base_dir, training_args.data_dir
        )
    if not os.path.isabs(training_args.logging_dir):
        training_args.logging_dir = os.path.join(
            training_args.base_dir, training_args.logging_dir
        )
    return training_args


def find_newest_checkpoint(checkpoint_path):
    # see if checkpoint_path's child contains pt or safetensors or pth
    if os.path.isdir(checkpoint_path) and any(
        x.endswith(("pt", "safetensors", "pth")) for x in os.listdir(checkpoint_path)
    ):
        return checkpoint_path

    else:
        return get_last_checkpoint(checkpoint_path)


def seed_everything(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
