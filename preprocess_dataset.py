# preprocess_dataset.py

import os
import glob
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np

# panoptic_segmentation_helper.py의 함수들을 사용하기 위해 import
# (또는 이 파일의 함수들을 복사/붙여넣기)
# 여기서는 효율성을 위해 panoptic_segmentation_helper.py의
# get_panoptic_masks_huggingface 코드를 가져와 일부 수정하여 사용합니다.

try:
    from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation
    from transformers.models.oneformer.processing_oneformer import OneFormerProcessor
    HF_AVAILABLE = True
except ImportError:
    print("Hugging Face Transformers가 설치되지 않았습니다.")
    print("pip install transformers timm")
    HF_AVAILABLE = False

def initialize_oneformer(model_name="shi-labs/oneformer_coco_swin_large", num_queries=150):
    """
    OneFormer 모델과 프로세서를 한 번만 로드하는 헬퍼 함수
    """
    if not HF_AVAILABLE:
        return None, None, None
        
    try:
        processor = OneFormerProcessor.from_pretrained(model_name)
        model = AutoModelForUniversalSegmentation.from_pretrained(model_name)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # 쿼리 수를 조정하여 더 많은/적은 객체를 탐지하도록 설정
        if hasattr(model.config, 'num_queries'):
            model.config.num_queries = num_queries
        
        return model, processor, device
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return None, None, None

@torch.no_grad()
def get_panoptic_object_count(
    image_pil: Image.Image, 
    model: torch.nn.Module, 
    processor: OneFormerProcessor, 
    device: str, 
    min_area: int = 100, 
    confidence_thresh: float = 0.5
) -> int:
    """
    OneFormer를 사용해 이미지의 객체 수(k)를 반환합니다.
    [panoptic_segmentation_helper.py의 get_panoptic_masks_huggingface 기반]
    
    Args:
        min_area: 객체로 간주할 최소 픽셀 면적
        confidence_thresh: 객체 탐지 신뢰도 임계값
    """
    try:
        inputs = processor(image_pil, ["panoptic"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        
        # 후처리 - 신뢰도 임계값(threshold) 적용
        predicted_panoptic_map = processor.post_process_panoptic_segmentation(
            outputs, 
            target_sizes=[image_pil.size[::-1]],
            threshold=confidence_thresh,
        )[0]
        
        panoptic_seg = predicted_panoptic_map["segmentation"].cpu().numpy()
        segments_info = predicted_panoptic_map["segments_info"]
        
        k = 0
        for segment in segments_info:
            segment_id = segment["id"]
            mask = (panoptic_seg == segment_id)
            
            # 최소 면적 조건을 만족하는 유효한 객체만 카운트
            if mask.sum() > min_area:
                k += 1
        return k
        
    except Exception as e:
        print(f"Panoptic segmentation 실패: {e}")
        return 0

def main(args):
    if not HF_AVAILABLE:
        print("필수 라이브러리가 없어 스크립트를 종료합니다.")
        return

    # 1. 모델과 프로세서를 한 번만 로드
    print(f"OneFormer 모델 로딩 중 (num_queries={args.num_queries})...")
    model, processor, device = initialize_oneformer(num_queries=args.num_queries)
    if model is None:
        return
    print(f"모델 로딩 완료. {device}에서 실행.")

    # 2. 이미지 디렉토리에서 모든 이미지 파일 검색
    image_paths = []
    for ext in args.ext:
        image_paths.extend(
            glob.glob(os.path.join(args.image_dir, "**", f"*.{ext}"), recursive=True)
        )
    
    print(f"총 {len(image_paths)}개의 이미지 파일을 찾았습니다.")

    # 3. JSONL 출력 파일 열기
    with open(args.output_file, 'w') as f_out:
        for path in tqdm(image_paths, desc="이미지 전처리 중"):
            try:
                # 4. 이미지 열기
                image = Image.open(path).convert("RGB")
                
                # 5. 객체 수 'k' 계산
                k = get_panoptic_object_count(
                    image, 
                    model, 
                    processor, 
                    device, 
                    min_area=args.min_area, 
                    confidence_thresh=args.confidence_thresh
                )
                
                # 6. k > 0 인 경우에만 데이터 저장
                if k > 0:
                    # 데이터셋의 이식성을 위해 절대 경로 대신 상대 경로 저장
                    relative_path = os.path.relpath(path, args.image_dir)
                    
                    data = {
                        "image_path": relative_path,
                        "object_count_k": k
                    }
                    f_out.write(json.dumps(data) + "\n")
                    
            except Exception as e:
                print(f"이미지 {path} 처리 중 오류 발생 (건너뜀): {e}")

    print(f"전처리 완료. 결과가 {args.output_file}에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="이미지 데이터셋을 전처리하여 객체 수(k)를 JSONL 파일로 저장합니다.")
    
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="전처리할 이미지들이 포함된 루트 디렉토리")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="결과 (image_path, k)를 저장할 JSONL 파일 경로")
    parser.add_argument("--ext", nargs="+", default=["jpg", "jpeg", "png", "webp"], 
                        help="처리할 이미지 확장자 (여러 개 가능)")
    
    # 세그멘테이션 하이퍼파라미터
    parser.add_argument("--min_area", type=int, default=1000, 
                        help="객체로 간주할 최소 픽셀 면적 (값이 클수록 큰 객체만 카운트)")
    parser.add_argument("--confidence_thresh", type=float, default=0.5, 
                        help="객체 탐지 신뢰도 임계값 (값이 높을수록 확실한 객체만 카운트)")
    parser.add_argument("--num_queries", type=int, default=150, 
                        help="OneFormer 모델의 최대 탐지 객체 수 (쿼리 수)")

    args = parser.parse_args()
    main(args)