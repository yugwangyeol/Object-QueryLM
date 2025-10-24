# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
from dataclasses import dataclass, field

import PIL.Image
import yaml
import torch
import transformers
import wandb
from transformers.trainer_utils import get_last_checkpoint
import datasets

from models.metaquery import MetaQueryConfig, MetaQuery
from trainer import MetaQueryTrainer, MetaQueryCallback
from trainer_utils import possible_override_args, find_newest_checkpoint, get_full_dirs
from dataset import get_train_datasets
from accelerate.utils import release_memory

datasets.disable_caching()
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_PROJECT"] = "MetaQuery" # W&B 프로젝트 이름 (필요시 수정)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from PIL import PngImagePlugin

PIL.Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class OverrideArguments:
    config_file: str = None # YAML 설정 파일 경로


@dataclass
class ModelArguments:
    # --- 기존 ModelArguments 필드들 유지 ---
    _gradient_checkpointing: bool = True
    vae_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    in_channels: int = 32
    vae_downsample_f: int = 32
    noise_scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    mllm_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    diffusion_model_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    loss_type: str = "flow"
    num_metaqueries: int = 64 # .yaml 파일에서 덮어쓸 수 있음
    modules_to_freeze: tuple[str] = ()
    modules_to_unfreeze: tuple[str] = ()
    max_input_text_tokens: int = 256
    connector_num_hidden_layers: int = 24
    system_prompt: str = "" # 이미지 전용이므로 시스템 프롬프트 비움 (필요시 설정)
    lambda_length: float = 0.1 # --- [ 신규 추가 ] --- Loss 가중치 추가 --- [ 추가 종료 ] ---
    inference_threshold: float = 0.5 # --- [ 신규 추가 ] --- 추론 임계값 추가 --- [ 추가 종료 ] ---


@dataclass
class DataArguments:
    # --- [ 수정 시작: 기본 데이터셋 변경 및 신규 인자 추가 ] ---
    train_datasets: dict[str, float] = field(
        default_factory=lambda: {
            "object_recon": -1, # 기본 학습 데이터셋을 object_recon으로 설정
        }
    )
    eval_dataset: str = "object_recon" # 기본 평가 데이터셋을 object_recon으로 설정
    target_image_size: int = 512

    # 1. 객체당 벡터 수 (j) - .yaml 파일에서 읽어옴
    j_vectors_per_object: int = 8
    # 2. 전처리된 데이터셋 JSONL 파일 경로 - .yaml 파일에서 읽어옴
    dataset_jsonl_path: str = "/path/to/placeholder/dataset.jsonl" # 기본값 설정
    # 3. 원본 이미지 파일들이 있는 기본 디렉토리 경로 - .yaml 파일에서 읽어옴
    image_base_dir: str = "/path/to/placeholder/images" # 기본값 설정
    # --- [ 수정 종료 ] ---


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # --- 기존 TrainingArguments 필드들 유지 (일부 기본값 수정 가능) ---
    base_dir: str = "/path/to/base_dir" # 실제 경로로 수정 필요
    output_dir: str = "output" # 기본 출력 디렉토리
    data_dir: str = ".cache" # 데이터 캐시 디렉토리
    logging_dir: str = "logs" # 로깅 디렉토리
    eval_on_start: bool = True
    eval_strategy: str = "steps"
    eval_steps: int = 1000 # 평가 간격 (조절 가능)
    eval_delay: int = 0
    per_device_train_batch_size: int = 32 # 배치 크기 (GPU 메모리에 맞게 조절)
    per_device_eval_batch_size: int = 4 # 평가 배치 크기 (조절 가능)
    gradient_accumulation_steps: int = 1
    optim: str = "adamw_torch"
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.5
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr": 1e-5})
    warmup_steps: int = 5000 # 워밍업 스텝 (조절 가능)
    logging_steps: int = 10 # 로깅 간격 (조절 가능)
    save_steps: int = 1000 # 저장 간격 (조절 가능)
    save_total_limit: int = 1 # 체크포인트 최대 저장 개수
    restore_callback_states_from_checkpoint: bool = True
    seed: int = 42
    data_seed: int = 42
    bf16: bool = True # BF16 사용 (지원 GPU 필요)
    tf32: bool = True # TF32 사용 (지원 GPU 필요)
    dataloader_num_workers: int = 4 # 데이터 로더 워커 수 (CPU 코어 수 고려)
    datasets_num_proc: int = os.getenv("OMP_NUM_THREADS", 12) # 데이터셋 전처리 프로세스 수
    dataloader_persistent_workers: bool = False
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    remove_unused_columns: bool = False # False로 유지 (dataset.py에서 처리)
    run_name: str = "object_recon_nj8_test" # 실행 이름 (wandb 등에 사용)
    report_to: str = "wandb" # 로깅 대상 (wandb 또는 none)
    ddp_find_unused_parameters: bool = False
    overwrite_output_dir: bool = False
    resume_from_checkpoint: str = None # 체크포인트 경로 (이어서 학습 시 지정)
    # --- [ 삭제 시작 ] --- (기존 num_train_epochs 삭제 - max_steps로 관리 권장)
    # num_train_epochs: float = 10.0
    # --- [ 삭제 종료 ] ---
    max_steps: int = 100000 # --- [ 신규 추가 ] --- 총 학습 스텝 수 지정 --- [ 추가 종료 ] ---


    def __post_init__(self):
        # --- [ 수정 시작: get_full_dirs 호출 위치 변경 ] ---
        # possible_override_args가 호출되기 전에 절대 경로 변환 수행
        try:
             # 절대 경로 처리 (base_dir 기준)
             self = get_full_dirs(self)
        except Exception as exc:
             print(f"Failed to process directories: {exc}")

        # 설정 파일(.yaml) 값으로 인자 덮어쓰기
        try:
            # OverrideArguments는 TrainingArguments보다 먼저 처리되어야 할 수 있음
            # HfArgumentParser가 처리 순서를 관리하므로 여기서 직접 호출 불필요
            pass # possible_override_args 호출은 HfArgumentParser가 처리
        except (FileNotFoundError, yaml.YAMLError) as exc:
            print(f"Failed to load override config: {exc}")
        # --- [ 수정 종료 ] ---
        super().__post_init__()


if __name__ == "__main__":
    # 1. 인자 파싱 설정 (OverrideArguments를 먼저 파싱하여 config_file 경로 확보)
    # HfArgumentParser는 dataclass 순서대로 인자를 파싱하고 덮어씀
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, OverrideArguments) # OverrideArguments 순서 변경 가능성 있음
    )
    # --- [ 수정 시작: 파싱 방식 변경 ] ---
    # parse_args_into_dataclasses 사용 시 remaining_strings 처리 불필요
    model_args, data_args, training_args, override_args = parser.parse_args_into_dataclasses()

    # 설정 파일(.yaml)이 있으면 해당 값으로 인자들(model_args, data_args, training_args)을 덮어씀
    # possible_override_args 함수는 모든 dataclass 객체를 받아 처리하도록 수정 필요
    model_args, data_args, training_args = possible_override_args(override_args, model_args, data_args, training_args)
    # --- [ 수정 종료 ] ---


    # --- [ 삭제 시작 ] --- (이미 __post_init__에서 처리)
    # # 절대 경로 처리 (base_dir 기준)
    # training_args = get_full_dirs(training_args)
    # --- [ 삭제 종료 ] ---

    # 입력 크기 계산 (기존 코드 유지)
    assert (
        data_args.target_image_size % model_args.vae_downsample_f == 0
    ), f"Image size must be divisible by {model_args.vae_downsample_f}"
    input_size = data_args.target_image_size // model_args.vae_downsample_f

    # 모델 로드 (기존 코드 유지, **model_args.__dict__ 전달 확인)
    if training_args.resume_from_checkpoint is not None:
        training_args.resume_from_checkpoint = find_newest_checkpoint(
            training_args.resume_from_checkpoint
        )
        model = MetaQuery.from_pretrained(
            training_args.resume_from_checkpoint,
            input_size=input_size,
            ignore_mismatched_sizes=True, # 중요: 헤드 크기가 다를 수 있으므로 True 유지
            **model_args.__dict__, # 수정된 model_args 전달
        )
    else:
        model = MetaQuery(
            config=MetaQueryConfig(
                input_size=input_size,
                **model_args.__dict__, # 수정된 model_args 전달
            ),
        )

    # 데이터셋 로드 및 전처리 (기존 코드 유지, get_train_datasets 호출 확인)
    with training_args.main_process_first(local=False):
        train_dataset, eval_dataset, gt_images, collate_fn = get_train_datasets(
            data_args, # 수정된 data_args 전달
            training_args,
            model_args,
            model.get_tokenize_fn(),
            model.get_tokenizer(),
        )

    # 트레이너 생성 (기존 코드 유지)
    trainer = MetaQueryTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[MetaQueryCallback()], # MetaQueryCallback 사용 확인
    )
    # 평가용 Ground Truth 이미지 로깅 (기존 코드 유지)
    if trainer.is_world_process_zero() and gt_images: # gt_images 비어있는지 확인
        try:
            trainer.log_images({"eval_ground_truth": [wandb.Image(image) for image in gt_images]})
        except Exception as e:
            print(f"Wandb Image logging failed: {e}")


    # 출력 디렉토리 설정 및 정보 출력 (기존 코드 유지)
    training_args.output_dir = str(
        os.path.join(training_args.output_dir, training_args.run_name)
    )
    if trainer.is_world_process_zero():
        if training_args.overwrite_output_dir and os.path.exists(
            training_args.output_dir
        ):
            print(f"Overwriting output directory: {training_args.output_dir}")
            shutil.rmtree(training_args.output_dir)
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Evaluation dataset size: {len(eval_dataset)}") # 평가 데이터셋 크기 출력 추가

    # 학습 루프 (기존 코드 유지)
    # --- [ 수정 시작: 학습 루프 조건 변경 ] ---
    # Epoch 대신 max_steps 기준으로 변경
    # while 루프 제거, trainer.train() 호출만 남김
    # --- [ 수정 종료 ] ---
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
             # Raise error if output dir exists but no checkpoint found and not overwriting
             # Or set resume_from_checkpoint=False? Let's assume user wants to resume if dir exists.
             print(f"Warning: Output directory {training_args.output_dir} exists but no checkpoint found.")
             # Decide behavior: raise error, set resume=False, or try to continue?
             # For safety, let's assume if checkpoint exists, we resume. If not, start fresh.
        elif last_checkpoint is not None:
             print(f"Resuming from checkpoint: {last_checkpoint}")

    # trainer.train() 호출 (resume_from_checkpoint 인자 사용)
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # 학습 완료 후 결과 저장 및 출력 (선택적)
    if trainer.is_world_process_zero():
        print("Training finished.")
        # Save final model state, metrics etc.
        trainer.save_model() # Save the final model
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()