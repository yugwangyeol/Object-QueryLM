# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset, Image
import PIL
import io
import torch
from torchvision.transforms import v2
import random
from torch.utils.data.dataset import ConcatDataset
from functools import partial
import os

def delete_keys_except(batch, except_keys):
    keys_to_delete = [key for key in list(batch.keys()) if key not in except_keys]
    for key in keys_to_delete:
        del batch[key]
    return batch


def _i2i_process_fn(batch, target_transform):
    images = batch["image"]
    captions = ["" for _ in range(len(images))]
    for i in range(len(images)):
        try:
            images[i] = PIL.Image.open(
                io.BytesIO(images[i]["bytes"])
                if images[i]["bytes"] is not None
                else images[i]["path"]
            ).convert("RGB")
        except:
            images[i] = None
            captions[i] = ""

    batch["target"] = [
        target_transform(image) if image is not None else None for image in images
    ]
    rand_probs = torch.rand((len(images), 1))
    null_image_mask = rand_probs <= 0.1
    images = [
        (
            PIL.Image.new("RGB", (image.width, image.height))
            if (image is not None and null_image_mask[i])
            else image
        )
        for i, image in enumerate(images)
    ]
    batch["caption"], batch["input_images"] = captions, [
        [image] if image is not None else None for image in images
    ]
    delete_keys_except(batch, ["target", "input_images", "caption"])
    return batch


def i2i_eval_process_fn(batch):
    images = batch["image"]
    captions = ["" for _ in range(len(images))]
    batch["caption"], batch["input_images"] = captions, [
        [image] if image is not None else None for image in images
    ]
    delete_keys_except(batch, ["input_images", "caption"])
    return batch


def _t2i_process_fn(batch, target_transform):
    images = batch["image"]
    captions = batch["caption"]
    captions = ["" if caption is None else caption for caption in captions]
    for i in range(len(images)):
        try:
            images[i] = PIL.Image.open(
                io.BytesIO(images[i]["bytes"])
                if images[i]["bytes"] is not None
                else images[i]["path"]
            ).convert("RGB")
        except:
            images[i] = None
            captions[i] = ""

    batch["target"] = [
        target_transform(image) if image is not None else None for image in images
    ]
    rand_probs = torch.rand((len(images), 1))
    null_caption_mask = rand_probs < 0.1
    captions = [
        caption if not null_caption_mask[i] else ""
        for i, caption in enumerate(captions)
    ]
    batch["caption"] = captions
    delete_keys_except(batch, ["target", "caption"])
    return batch


def t2i_eval_process_fn(batch):
    captions = batch["caption"]
    batch["caption"] = captions
    delete_keys_except(batch, ["caption"])
    return batch


def _inst_process_fn(batch, target_transform):
    source_images = batch["source_images"]
    caption = batch["caption"]
    rand_probs = torch.rand((len(batch["target_image"]), 1))
    null_caption_mask = rand_probs < 0.2
    null_image_mask = (rand_probs >= 0.1) & (rand_probs < 0.3)
    caption = [
        caption if not null_caption_mask[i] else "" for i, caption in enumerate(caption)
    ]
    source_images = (
        [
            (
                image
                if not null_image_mask[i]
                else [PIL.Image.new("RGB", (img.width, img.height)) for img in image]
            )
            for i, image in enumerate(source_images)
        ]
        if source_images is not None
        else None
    )
    batch["caption"], batch["input_images"] = caption, source_images
    batch["target"] = [
        target_transform(img.convert("RGB")) for img in batch["target_image"]
    ]
    delete_keys_except(batch, ["target", "input_images", "caption"])
    return batch


def inst_eval_process_fn(batch):
    source_images = batch["source_images"]
    caption = batch["caption"]
    batch["caption"], batch["input_images"] = caption, source_images
    delete_keys_except(batch, ["caption", "input_images"])
    return batch


def _editing_process_fn(batch, target_transform, ground_truth_transform):
    source_images = batch["source_image"]
    target_images = batch["target_image"]
    captions = batch["caption"]
    captions = ["" if caption is None else caption[-1] for caption in captions]
    for i in range(len(source_images)):
        try:
            source_images[i] = PIL.Image.open(
                io.BytesIO(source_images[i]["bytes"])
                if source_images[i]["bytes"] is not None
                else source_images[i]["path"]
            ).convert("RGB")
            target_images[i] = PIL.Image.open(
                io.BytesIO(target_images[i]["bytes"])
                if target_images[i]["bytes"] is not None
                else target_images[i]["path"]
            ).convert("RGB")
        except:
            source_images[i] = None
            target_images[i] = None
            captions[i] = ""

    batch["target"] = [
        target_transform(image) if image is not None else None
        for image in target_images
    ]
    rand_probs = torch.rand((len(target_images), 1))
    null_image_mask = rand_probs <= 0.1
    source_images = [
        (
            PIL.Image.new("RGB", (image.width, image.height))
            if (image is not None and null_image_mask[i])
            else image
        )
        for i, image in enumerate(source_images)
    ]
    batch["caption"], batch["input_images"] = captions, [
        [image] if image is not None else None for image in source_images
    ]
    delete_keys_except(batch, ["target", "input_images", "caption"])
    return batch


def editing_eval_process_fn(batch):
    source_images = batch["source_image"]
    captions = batch["caption"]
    captions = ["" if caption is None else caption[-1] for caption in captions]
    batch["caption"], batch["input_images"] = captions, [
        [image] if image is not None else None for image in source_images
    ]
    delete_keys_except(batch, ["input_images", "caption"])
    return batch

# --- [ 신규 추가 ] ---
# 이미지 전용, 객체 재구성 학습을 위한 데이터 처리 함수
def _object_recon_process_fn(batch, target_transform):
    images = batch["image"] # JSONL에서 image_path 대신 image로 로드됨
    # object_count_n은 batch에 이미 존재한다고 가정
    
    for i in range(len(images)):
        #print(f"Loading image from path: {images[i]}")  # 디버그 출력
        try:
            # 이미지가 경로 형태일 수 있으므로 PIL로 열기 시도
            if isinstance(images[i], str): 
                images[i] = PIL.Image.open(images[i]).convert("RGB")
            # 이미 BytesIO 형태일 경우 처리
            elif isinstance(images[i], dict) and 'bytes' in images[i]:
                 images[i] = PIL.Image.open(
                    io.BytesIO(images[i]["bytes"])
                ).convert("RGB")
            elif not isinstance(images[i], PIL.Image.Image):
                 raise ValueError("Unsupported image format")
        except Exception as e:
            print(f"이미지 로딩 오류 (건너뜀): {e}")
            images[i] = None

    # 원본 이미지를 target으로 변환
    batch["target"] = [
        target_transform(image) if image is not None else None for image in images
    ]
    
    # 캡션은 항상 빈 문자열로 설정 (이미지 전용 학습)
    batch["caption"] = ["" for _ in range(len(images))] 
    # 입력 이미지 리스트 생성 (각 이미지 하나를 리스트로 감쌈)
    batch["input_images"] = [
        [image] if image is not None else None for image in images
    ]
    
    # 필요한 키("target", "input_images", "caption", "object_count_n")만 남김
    delete_keys_except(batch, ["target", "input_images", "caption", "object_count_n"])
    return batch
# --- [ 추가 종료 ] ---


def _collate_fn(batch, tokenize_func, tokenizer, data_args):
    # None인 타겟(이미지 로딩 실패 등) 필터링
    none_idx = [i for i, example in enumerate(batch) if example["target"] is None]
    if len(none_idx) > 0:
        batch = [example for i, example in enumerate(batch) if i not in none_idx]
        # 만약 필터링 후 배치가 비어있으면 빈 딕셔너리 반환 (오류 방지)
        if not batch:
             return {} 
    
    # target 텐서 스택
    return_dict = {"target": torch.stack([example["target"] for example in batch])}

    # --- [ 신규 추가: n*j 로직 ] ---
    # "object_count_n" 키가 있는지 확인 (우리 전용 데이터셋)
    # batch가 비어있지 않고, 첫번째 요소에 키가 있는지 확인
    if batch and "object_count_n" in batch[0]:
        # n (객체 수)에 j (객체당 벡터 수)를 곱하여 k (총 벡터 수)를 계산
        total_k_vectors = [
            example["object_count_n"] * data_args.j_vectors_per_object
            for example in batch
        ]
        # 모델에는 "object_count_k" 라는 키로 k 값을 전달
        return_dict["object_count_k"] = torch.tensor(total_k_vectors, dtype=torch.long)
    # --- [ 추가 종료 ] ---

    # 입력 이미지 처리
    input_images = [
        example["input_images"] if "input_images" in example else None
        for example in batch
    ]

    # 입력 캡션 (항상 ""(빈 문자열) 리스트가 됨)
    captions = [example["caption"] for example in batch]

    # 이미지가 있는 경우 토크나이징
    if any(input_images):
        (
            return_dict["input_ids"],
            return_dict["attention_mask"],
            return_dict["pixel_values"],
            return_dict["image_sizes"],
        ) = tokenize_func(
            tokenizer, captions, input_images
        )
    # 캡션만 있는 경우 (이 시나리오에서는 발생하지 않음)
    else:
        return_dict["input_ids"], return_dict["attention_mask"] = tokenize_func(
            tokenizer, captions
        )
    return return_dict


def get_train_datasets(data_args, training_args, model_args, tokenize_func, tokenizer):
    train_datasets = {}
    
    # --- [ 삭제 시작 ] ---
    # # 기존 cc12m_i2i 데이터셋 로드 로직 (필요 없으므로 주석 처리 또는 삭제)
    # if "cc12m_i2i" in data_args.train_datasets:
    #     # ... (기존 cc12m_i2i 로드 및 처리 로직) ...
    #     train_datasets["cc12m_i2i"] = train_dataset

    # # 기존 cc12m_t2i 데이터셋 로드 로직 (필요 없으므로 주석 처리 또는 삭제)
    # if "cc12m_t2i" in data_args.train_datasets:
    #     # ... (기존 cc12m_t2i 로드 및 처리 로직) ...
    #     train_datasets["cc12m_t2i"] = train_dataset
        
    # # 기존 inst2m 데이터셋 로드 로직 (필요 없으므로 주석 처리 또는 삭제)
    # if "inst2m" in data_args.train_datasets:
    #      # ... (기존 inst2m 로드 및 처리 로직) ...
    #     train_datasets["inst2m"] = train_dataset
        
    # # 기존 ominiedit 데이터셋 로드 로직 (필요 없으므로 주석 처리 또는 삭제)
    # if "ominiedit" in data_args.train_datasets:
    #     # ... (기존 ominiedit 로드 및 처리 로직) ...
    #     train_datasets["ominiedit"] = train_dataset
    # --- [ 삭제 종료 ] ---

    # --- [ 신규 추가 ] ---
    # 1. 새로운 object_recon 데이터셋 로드
    if "object_recon" in data_args.train_datasets:
        train_dataset = load_dataset(
            "json",
            # preprocess_dataset.py로 생성한 JSONL 파일 경로를 지정해야 합니다.
            # 예: "/data/my_dataset.jsonl"
            data_files={"train": data_args.dataset_jsonl_path},
            cache_dir=training_args.data_dir,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )
        
        # 테스트 실행 시 데이터셋 크기 제한
        if training_args.run_name == "test":
            train_dataset = train_dataset.select(range(10000)) # 예시로 10000개 선택

        # image_path 컬럼을 image 컬럼으로 변경 (Hugging Face Image 타입으로 캐스팅)
        # 이미지 디렉토리 경로와 결합하여 절대 경로 생성 필요
        def map_image_path(example):
            # 하드코딩된 경로 대신 data_args에서 이미지 기본 경로 읽어오기
            # os.path.join을 사용하여 OS 호환성 확보
            example['image'] = os.path.join(data_args.image_base_dir, example['image_path'])
            #print(f"Mapped image path: {example['image']}")  # 디버그 출력
            return example
            
        train_dataset = train_dataset.map(map_image_path, num_proc=training_args.datasets_num_proc)
        train_dataset = train_dataset.cast_column("image", Image(decode=False)) # decode=False로 설정하여 나중에 process_fn에서 열도록 함
        
        # 생성된 데이터셋을 train_datasets 딕셔너리에 추가
        train_datasets["object_recon"] = train_dataset
    # --- [ 추가 종료 ] ---

    # 타겟 이미지 변환 정의 (원본 이미지 -> VAE 입력)
    target_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True), # 고품질 리사이징
            v2.CenterCrop(data_args.target_image_size),
            v2.ToImage(), # PIL -> Tensor
            v2.ToDtype(torch.float32, scale=True), # 0-255 -> 0-1 float32
            v2.Normalize([0.5], [0.5]), # 0-1 -> -1~1 정규화
        ]
    )

    # 평가용 Ground Truth 이미지 변환 정의 (단순 리사이즈 및 크롭)
    ground_truth_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(data_args.target_image_size),
        ]
    )
    
    # --- [ 삭제 시작 ] ---
    # # 기존 process_fn 정의 (주석 처리 또는 삭제)
    # i2i_process_fn = partial(_i2i_process_fn, target_transform=target_transform)
    # t2i_process_fn = partial(_t2i_process_fn, target_transform=target_transform)
    # inst_process_fn = partial(_inst_process_fn, target_transform=target_transform)
    # editing_process_fn = partial(
    #     _editing_process_fn,
    #     target_transform=target_transform,
    #     ground_truth_transform=ground_truth_transform,
    # )
    # --- [ 삭제 종료 ] ---

    # --- [ 신규 추가 ] ---
    # 2. 새로운 처리 함수 등록
    object_recon_process_fn = partial(_object_recon_process_fn, target_transform=target_transform)
    # --- [ 추가 종료 ] ---

    # collate 함수 정의
    collate_fn = partial(_collate_fn, tokenize_func=tokenize_func, tokenizer=tokenizer, data_args=data_args)

    # 평가 데이터셋 설정
    # data_args.eval_dataset이 train_datasets에 정의된 이름과 일치해야 함
    if data_args.eval_dataset not in train_datasets:
        raise ValueError(f"eval_dataset '{data_args.eval_dataset}' not found in train_datasets: {list(train_datasets.keys())}")
        
    eval_dataset = train_datasets[data_args.eval_dataset].select(
        # 평가 데이터셋 크기 제한 (예: 100개) - 필요에 따라 조절
        range(min(100, len(train_datasets[data_args.eval_dataset]))) 
    )
    
    # 평가 데이터셋의 이미지 컬럼을 즉시 디코딩하도록 캐스팅 (메모리 사용량 증가 가능)
    eval_dataset = eval_dataset.cast_column("image", Image(decode=True)) 
    
    # Ground Truth 이미지 추출 및 변환
    gt_images = [
        ground_truth_transform(image.convert("RGB")) for image in eval_dataset["image"]
    ]

    # --- [ 삭제 시작 ] ---
    # # 기존 평가 데이터셋 변환 로직 (주석 처리 또는 삭제)
    # if data_args.eval_dataset in ["cc12m_i2i"]:
    #     eval_dataset.set_transform(i2i_eval_process_fn)
    # elif data_args.eval_dataset in ["cc12m_t2i"]:
    #     eval_dataset.set_transform(t2i_eval_process_fn)
    # elif data_args.eval_dataset in ["inst2m"]:
    #     eval_dataset.set_transform(inst_eval_process_fn)
    # elif data_args.eval_dataset in ["ominiedit"]:
    #     eval_dataset.set_transform(editing_eval_process_fn)
    # else:
    #     raise ValueError(f"Unknown eval_dataset: {data_args.eval_dataset}")
    # --- [ 삭제 종료 ] ---
    
    # --- [ 신규 추가 ] ---
    # 3. 새로운 처리 함수 적용
    if data_args.eval_dataset == "object_recon":
         # 평가 시에는 이미지 경로 대신 바로 이미지를 사용하도록 처리하는 함수 필요
         # 간단하게는 collate_fn과 유사한 로직을 사용하거나, 
         # object_recon_process_fn에서 object_count_n 키를 제거하는 평가용 함수를 만들 수 있음
         # 여기서는 object_recon_process_fn을 그대로 사용 (단, object_count_n은 사용 안됨)
         eval_dataset.set_transform(object_recon_process_fn) 
    else:
        # 다른 평가 데이터셋이 필요하면 여기에 추가
         raise ValueError(f"Unsupported eval_dataset for this setup: {data_args.eval_dataset}")
    # --- [ 추가 종료 ] ---

    # 학습 데이터셋에 변환 함수 적용 및 셔플링
    for dataset_name, train_ds in train_datasets.items():
        # --- [ 삭제 시작 ] ---
        # # 기존 데이터셋 변환 로직 (주석 처리 또는 삭제)
        # if dataset_name in ["cc12m_i2i"]:
        #     # ...
        # elif dataset_name in ["cc12m_t2i"]:
        #     # ...
        # elif dataset_name in ["inst2m"]:
        #     # ...
        # elif dataset_name in ["ominiedit"]:
        #     # ...
        # --- [ 삭제 종료 ] ---
        
        # --- [ 신규 추가 ] ---
        if dataset_name == "object_recon":
            # 이미 Image(decode=False)로 캐스팅 되었으므로 추가 캐스팅 불필요
            train_ds.set_transform(object_recon_process_fn)
        # --- [ 추가 종료 ] ---
        else:
            # 다른 학습 데이터셋이 필요하면 여기에 추가
            raise ValueError(f"Unsupported train_dataset for this setup: {dataset_name}")
            
        # 데이터셋 셔플링
        train_datasets[dataset_name] = train_ds.shuffle(seed=training_args.data_seed)

    # 최종 학습 데이터셋 생성 (현재는 object_recon 하나만 있음)
    if len(train_datasets) > 1:
        # 여러 데이터셋을 합쳐야 하는 경우 ConcatDataset 사용 (여기서는 불필요)
        final_train_dataset = ConcatDataset(list(train_datasets.values()))
    elif len(train_datasets) == 1:
        final_train_dataset = train_datasets[list(train_datasets.keys())[0]]
    else:
         raise ValueError("No training dataset loaded!")

    return final_train_dataset, eval_dataset, gt_images, collate_fn