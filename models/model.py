# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Tuple # --- [ 수정 시작 ] --- Tuple 임포트 추가 --- [ 수정 종료 ] ---

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as v2

from transformers import PretrainedConfig, PreTrainedModel, AutoProcessor
from transformers import (
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Config,
)

from diffusers.models.normalization import RMSNorm
from diffusers import SanaTransformer2DModel, UNet2DConditionModel

from models.transformer_encoder import Qwen2Encoder


class MLLMInContextConfig(PretrainedConfig):
    model_type = "mllm-in-context"

    def __init__(
        self,
        mllm_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        diffusion_model_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        in_channels: int = 32,
        input_size: int = 32,
        num_metaqueries: int = 64,
        _gradient_checkpointing: bool = True,
        max_input_text_tokens: int = 256,
        connector_num_hidden_layers: int = 24,
        system_prompt: str = "You will be given an image or its caption. Please describe the content of the image in detail in your own words.",
        **kwargs,
    ):
        super().__init__()
        self.mllm_id = mllm_id
        self.diffusion_model_id = diffusion_model_id
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_metaqueries = num_metaqueries
        self._gradient_checkpointing = _gradient_checkpointing
        self.max_input_text_tokens = max_input_text_tokens
        self.connector_num_hidden_layers = connector_num_hidden_layers
        self.system_prompt = system_prompt


class MLLMInContext(PreTrainedModel):
    config_class = MLLMInContextConfig

    def __init__(
        self,
        config: MLLMInContextConfig,
    ) -> None:
        super().__init__(config)
        self._gradient_checkpointing = config._gradient_checkpointing
        self.config = config
        if "Qwen2.5-VL" in config.mllm_id:
            self.mllm_type = "qwenvl"
        elif "Qwen" in config.mllm_id:
            self.mllm_type = "qwenlm"
        elif "Llama" in config.mllm_id:
            self.mllm_type = "llamaml"
        else:
            self.mllm_type = "llavaov"

        if self.mllm_type == "llavaov":
            self.mllm_backbone = LlavaOnevisionForConditionalGeneration.from_pretrained(
                config.mllm_id, attn_implementation="sdpa"
            )
            self.mllm_backbone.language_model.config.use_sliding_window = False
            self.mllm_backbone.language_model.config.sliding_window = None
            num_embeddings = self.mllm_backbone.get_input_embeddings().num_embeddings
            self.num_embeddings = num_embeddings
            if config.num_metaqueries > 0:
                try:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2
                    )
                except:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2, mean_resizing=False
                    )

            def freeze_hook(grad):
                if grad is not None: # --- [ 신규 추가 ] --- grad가 None일 경우 방지 --- [ 추가 종료 ] ---
                    grad[: self.num_embeddings].zero_()
                return grad

            self.mllm_backbone.language_model.model.embed_tokens.weight.register_hook(
                freeze_hook
            )
            self.mllm_hidden_size = self.mllm_backbone.config.text_config.hidden_size
            self.mllm_backbone.language_model.lm_head = nn.Identity()

            self.tokenizer = AutoProcessor.from_pretrained(config.mllm_id)
            self.tokenizer.tokenizer.padding_side = "left"
            self.tokenizer.resize_fn = v2.Compose([v2.Resize(384), v2.CenterCrop(384)])
            # 0.5B 896

        elif self.mllm_type == "qwenvl":
            self.mllm_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.mllm_id, attn_implementation="sdpa"
            )
            self.mllm_backbone.model.config.use_sliding_window = False
            self.mllm_backbone.model.config.sliding_window = None
            num_embeddings = self.mllm_backbone.get_input_embeddings().num_embeddings
            self.num_embeddings = num_embeddings
            if config.num_metaqueries > 0:
                try:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2
                    )
                except:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2, mean_resizing=False
                    )

            def freeze_hook(grad):
                if grad is not None: # --- [ 신규 추가 ] --- grad가 None일 경우 방지 --- [ 추가 종료 ] ---
                    grad[: self.num_embeddings].zero_()
                return grad

            self.mllm_backbone.model.embed_tokens.weight.register_hook(freeze_hook)
            self.mllm_hidden_size = self.mllm_backbone.config.hidden_size
            self.mllm_backbone.lm_head = nn.Identity()

            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            self.tokenizer = AutoProcessor.from_pretrained(
                config.mllm_id, min_pixels=min_pixels, max_pixels=max_pixels
            )
            self.tokenizer.tokenizer.padding_side = "left"
            self.tokenizer.resize_fn = None
            # 3B 2048
            # 7B 3584

        else:
            raise ValueError(f"Unsupported model: {config.mllm_id}")

        self.tokenizer.mllm_type = self.mllm_type
        self.tokenizer.max_input_text_tokens = config.max_input_text_tokens
        self.tokenizer.num_metaqueries = config.num_metaqueries
        self.tokenizer.system_prompt = config.system_prompt
        self.pad_token_id = getattr(
            self.tokenizer, "tokenizer", self.tokenizer
        ).pad_token_id
        if config.num_metaqueries > 0:
            tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            # --- [ 삭제 시작 ] --- (기존 토큰 추가 로직은 불필요할 수 있음, 필요시 주석 해제)
            # tokenizer.add_special_tokens(
            #     {
            #         "additional_special_tokens": [
            #             f"<pad_token_{i}>"
            #             for i in range(num_embeddings - len(tokenizer))
            #         ]
            #     }
            # )
            # --- [ 삭제 종료 ] ---
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": ["<begin_of_img>", "<end_of_img>"]
                    + [f"<img{i}>" for i in range(self.tokenizer.num_metaqueries)]
                }
            )
            self.boi_token_id = tokenizer.convert_tokens_to_ids("<begin_of_img>")
            self.eoi_token_id = tokenizer.convert_tokens_to_ids("<end_of_img>")

        if "Sana" in config.diffusion_model_id:
            self.transformer = SanaTransformer2DModel.from_pretrained(
                config.diffusion_model_id,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            )
            input_scale = math.sqrt(5.5)
            # 2304 --> 2240

        elif "stable-diffusion-v1-5" in config.diffusion_model_id:
            self.transformer = UNet2DConditionModel.from_pretrained(
                config.diffusion_model_id, subfolder="unet"
            )
            input_scale = 1
            # 768

        else:
            raise ValueError(f"Unsupported model: {config.diffusion_model_id}")

        self.connector_in_dim = self.mllm_hidden_size
        self.connector_out_dim = (
            getattr(self.transformer.config, "caption_channels", None)
            or getattr(self.transformer.config, "encoder_hid_dim", None)
            or getattr(self.transformer.config, "cross_attention_dim", None)
        )

        norm = RMSNorm(self.connector_out_dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            norm.weight.fill_(input_scale)

        encoder = Qwen2Encoder(
            Qwen2Config(
                hidden_size=self.connector_in_dim,
                intermediate_size=self.connector_in_dim * 4,
                num_hidden_layers=config.connector_num_hidden_layers,
                num_attention_heads=self.connector_in_dim // 64,
                num_key_value_heads=self.connector_in_dim // 64,
                initializer_range=0.014,
                use_cache=False,
                rope=True,
                qk_norm=True,
            ),
        )
        self.connector = nn.Sequential(
            encoder,
            nn.Linear(self.connector_in_dim, self.connector_out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.connector_out_dim, self.connector_out_dim),
            norm,
        )

        # --- [ 신규 추가 START ] ---
        # "유효성/길이" 예측을 위한 두 번째 헤드 (간단한 Linear 레이어)
        self.objectness_head = nn.Linear(self.mllm_hidden_size, 1)
        # --- [ 신규 추가 END ] ---


        if config._gradient_checkpointing:
            try:
                self.mllm_backbone.gradient_checkpointing_enable(
                    {"use_reentrant": False}
                )
            except:
                pass
            if not isinstance(self.connector, nn.Identity):
                for module in self.connector:
                    if isinstance(module, Qwen2Encoder):
                        module.gradient_checkpointing_enable({"use_reentrant": False})
            self.transformer.enable_gradient_checkpointing()

    def get_tokenizer(self):
        return self.tokenizer

    def get_tokenize_fn(self):
        return self.tokenize

    def get_resize_fn(self):
        # --- [ 삭제 시작 ] --- (resize_fn은 tokenizer에서 처리하므로 삭제)
        # return self.resize_fn
        # --- [ 삭제 종료 ] ---
        return None # --- [ 신규 추가 ] --- 명시적으로 None 반환

    @staticmethod
    @torch.no_grad()
    def tokenize(
        tokenizer, caption, image=None, text_response=None, add_generation_prompt=True
    ):
        if not isinstance(caption, List):
            caption = [caption]

        prefix = (
            [
                {
                    "role": "system",
                    "content": (
                        tokenizer.system_prompt
                        if tokenizer.mllm_type == "qwenlm"
                        else [{"type": "text", "text": tokenizer.system_prompt}]
                    ),
                },
            ]
            if tokenizer.system_prompt is not None and tokenizer.system_prompt # --- [ 수정 시작 ] --- system_prompt가 비어있지 않은 경우만 추가 --- [ 수정 종료 ] ---
            else []
        )

        # --- [ 수정 시작 ] --- num_metaqueries가 0 이하일 때도 suffix 처리 --- [ 수정 종료 ] ---
        if not add_generation_prompt or tokenizer.num_metaqueries <= 0:
            suffix = "<|im_end|>" # MetaQuery 없을 때도 종료 토큰 필요
        else:
            suffix = (
                "\n<begin_of_img>"
                + "".join([f"<img{i}>" for i in range(tokenizer.num_metaqueries)])
                + "<end_of_img><|im_end|>"
            )

        caption = [
            tokenizer.decode(
                tokenizer(text=cap, return_tensors="pt", padding=False).input_ids[
                    0, : tokenizer.max_input_text_tokens
                ]
            ) if cap else "" # --- [ 수정 시작 ] --- 빈 캡션 처리 추가 --- [ 수정 종료 ] ---
            for cap in caption
        ]
        if image is not None:
            # If image is not a list, wrap it in a list
            if not isinstance(image, list):
                image = [image]
            # If each batch item is not a list, wrap it in a single-element list (or empty list if None)
            for i, img_list in enumerate(image): # --- [ 수정 시작 ] --- 변수명 명확화 --- [ 수정 종료 ] ---
                if img_list and not isinstance(img_list, list):
                    image[i] = [img_list] # --- [ 수정 시작 ] --- img -> img_list --- [ 수정 종료 ] ---

            # Resize each image in each batch if resize_fn is not None
            if tokenizer.resize_fn is not None:
                image = [
                    [tokenizer.resize_fn(sub_img.convert("RGB")) for sub_img in imgs] if imgs else None # --- [ 수정 시작 ] --- convert 추가 --- [ 수정 종료 ] ---
                    for imgs in image
                ]

            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": (
                            # --- [ 수정 시작 ] --- 이미지 토큰과 텍스트 토큰 분리 (LLaVA 스타일) --- [ 수정 종료 ] ---
                            ([{"type": "image"}] * len(imgs) if imgs else []) # 이미지 토큰 먼저
                            + ([{"type": "text", "text": cap}] if cap else []) # 텍스트 토큰 나중
                        ),
                    },
                ]
                for cap, imgs in zip(caption, image)
            ]
            # Filter out None images before passing to tokenizer
            kwargs = {"images": [img for img_list in image if img_list for img in img_list]} # --- [ 수정 시작 ] --- 이미지 리스트 처리 수정 --- [ 수정 종료 ] ---

        elif tokenizer.mllm_type in ["qwenlm", "llamaml"]:
            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": cap,
                    },
                ]
                for cap in caption
            ]
            kwargs = dict()

        else: # llavaov or others (text only)
            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": cap}] if cap else [], # --- [ 수정 시작 ] --- 빈 캡션 처리 --- [ 수정 종료 ] ---
                    },
                ]
                for cap in caption
            ]
            kwargs = dict()

        prompts = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) # --- [ 수정 시작 ] --- tokenize=False, add_generation_prompt=False --- [ 수정 종료 ] ---
            for conv in conversations
        ]

        if text_response is not None:
            prompts = [p + t.strip() for p, t in zip(prompts, text_response)]

        # --- [ 수정 시작 ] --- suffix를 apply_chat_template 후에 추가 --- [ 수정 종료 ] ---
        if add_generation_prompt:
            try:
                assistant_prompt_marker = tokenizer.tokenizer.apply_chat_template([{'role': 'assistant', 'content': ''}], tokenize=False, add_generation_prompt=True)

                # BOS 토큰이 존재하고 문자열일 경우에만 replace 수행
                if tokenizer.tokenizer.bos_token is not None and isinstance(tokenizer.tokenizer.bos_token, str):
                    assistant_prompt_marker = assistant_prompt_marker.replace(tokenizer.tokenizer.bos_token, "") # Avoid double BOS

            except IndexError:
                assistant_prompt_marker = ""
                print("Warning: Could not automatically determine assistant marker. Check chat template.")

            prompts = [p + assistant_prompt_marker for p in prompts]

        # MetaQuery suffix 추가
        if tokenizer.num_metaqueries > 0:
             suffix_mq = (
                "\n<begin_of_img>"
                + "".join([f"<img{i}>" for i in range(tokenizer.num_metaqueries)])
                + "<end_of_img>" # Removed <|im_end|> here, add later if needed by tokenizer
            )
             prompts = [p + suffix_mq for p in prompts]

        # Add EOS/EOT token if necessary (tokenizer usually handles this during encoding)
        # Example for Qwen: Add <|im_end|> if not already added by template/suffix logic
        # if tokenizer.tokenizer.eos_token == '<|im_end|>':
        #    prompts = [p + '<|im_end|>' if not p.endswith('<|im_end|>') else p for p in prompts]

        # --- [ 수정 종료 ] ---
        text_inputs = tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            **kwargs,
        )

        if tokenizer.mllm_type == "qwenvl" and "pixel_values" in text_inputs:
            text_inputs["pixel_values"] = text_inputs["pixel_values"].unsqueeze(0)

        # --- [ 수정 시작 ] --- pixel_values가 없으면 None으로 명시적 반환 --- [ 수정 종료 ] ---
        return (
            text_inputs.input_ids,
            text_inputs.attention_mask,
            text_inputs.get("pixel_values"), # .get() 사용
            text_inputs.get("image_sizes") # .get() 사용
        )
        # --- [ 수정 종료 ] ---

    # --- [ 수정 시작 ] --- 반환 타입 튜플 명시 --- [ 수정 종료 ] ---
    def encode_condition(
        self, input_ids, attention_mask, pixel_values, image_sizes, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    # --- [ 수정 종료 ] ---
        if self.mllm_type == "llavaov":
            # --- [ 수정 시작 ] --- output_hidden_states=True 추가 --- [ 수정 종료 ] ---
            mllm_outputs = self.mllm_backbone(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # Use last hidden state instead of logits
            mllm_output_embeds = mllm_outputs.hidden_states[-1]
            # --- [ 수정 종료 ] ---
        elif self.mllm_type == "qwenvl":
            # --- [ 수정 시작 ] --- output_hidden_states=True 추가 --- [ 수정 종료 ] ---
            mllm_outputs = self.mllm_backbone(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_sizes,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # Use last hidden state instead of logits
            mllm_output_embeds = mllm_outputs.hidden_states[-1]
            # --- [ 수정 종료 ] ---
        elif self.mllm_type in ["qwenlm", "llamaml"]:
             # --- [ 수정 시작 ] --- output_hidden_states=True 추가 --- [ 수정 종료 ] ---
            mllm_outputs = self.mllm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            mllm_output_embeds = mllm_outputs.hidden_states[-1]
             # --- [ 수정 종료 ] ---
        else:
            raise ValueError(f"Unsupported model: {self.mllm_type}")

        if self.tokenizer.num_metaqueries > 0:
            # Get positions for all sequences in batch at once
            # --- [ 수정 시작 ] --- 토큰 ID 존재 여부 확인 추가 --- [ 수정 종료 ] ---
            boi_pos_indices = torch.where(input_ids == self.boi_token_id)
            eoi_pos_indices = torch.where(input_ids == self.eoi_token_id)

            # Check if both tokens are present in all batch items
            if not (len(boi_pos_indices[0]) == input_ids.shape[0] and len(eoi_pos_indices[0]) == input_ids.shape[0]):
                 # Handle cases where tokens might be missing (e.g., during text-only inference)
                 # Option 1: Return default values or raise error
                 # For now, let's assume they should always be present during training/relevant inference
                 # If this fails often, error handling or different logic is needed.
                 # We can return None for objectness_logits if metaqueries are not properly found
                 print("Warning: <begin_of_img> or <end_of_img> token not found in all batch items.")
                 # Fallback: Process the entire sequence through connector? Or return error/None?
                 # Let's try processing the whole sequence for now, but mark objectness as None
                 connector_embeds = self.connector(mllm_output_embeds)
                 return connector_embeds, None, attention_mask # Return None for objectness_logits

            boi_pos = boi_pos_indices[1]
            eoi_pos = eoi_pos_indices[1]
            # --- [ 수정 종료 ] ---


            # Create mask for selecting tokens between BOI and EOI
            batch_size, seq_len = input_ids.shape
            indices = torch.arange(seq_len, device=input_ids.device)[None, :].expand(
                batch_size, -1
            )
            # Ensure mask indices are correctly calculated for variable positions
            mask = (indices > boi_pos.unsqueeze(1)) & (indices < eoi_pos.unsqueeze(1)) # Use unsqueeze

            # --- [ 수정 시작 ] --- Handle potential empty masks --- [ 수정 종료 ] ---
            # Check if any mask is empty, which shouldn't happen if boi/eoi logic is correct
            if not mask.any(dim=1).all():
                print("Warning: Empty mask detected for some items in the batch.")
                # Handle this case, e.g., return None or default tensor
                # Let's return None for objectness_logits here too
                connector_embeds = self.connector(mllm_output_embeds)
                return connector_embeds, None, attention_mask
            # --- [ 수정 종료 ] ---


            # --- [ 수정 시작 ] --- Extract embeddings using gather or indexing --- [ 수정 종료 ] ---
            # Using masked_select might be cleaner if shapes are tricky
            # Ensure correct reshaping based on expected num_metaqueries
            # Example using indexing (might need adjustment based on exact mask behavior)
            # This part assumes a fixed number of tokens between boi and eoi, which might be wrong.
            # Let's revert to a safer method using padding or careful indexing if needed.

            # Alternative using list comprehension and stacking (safer but potentially slower)
            metaquery_embeds_list = []
            attention_mask_list = []
            max_len = 0
            for i in range(batch_size):
                 item_mask = mask[i]
                 embeds = mllm_output_embeds[i][item_mask]
                 attn_mask = attention_mask[i][item_mask]
                 metaquery_embeds_list.append(embeds)
                 attention_mask_list.append(attn_mask)
                 max_len = max(max_len, embeds.shape[0])

            # Pad sequences to max_len
            padded_embeds = []
            padded_attn_masks = []
            for embeds, attn_mask in zip(metaquery_embeds_list, attention_mask_list):
                 pad_len = max_len - embeds.shape[0]
                 padded_embeds.append(F.pad(embeds, (0, 0, 0, pad_len))) # Pad sequence dim
                 padded_attn_masks.append(F.pad(attn_mask, (0, pad_len), value=0)) # Pad attn mask

            metaquery_embeds = torch.stack(padded_embeds, dim=0) # Shape: [bsz, max_len, hidden_dim]
            final_attention_mask = torch.stack(padded_attn_masks, dim=0) # Shape: [bsz, max_len]

            # --- [ 수정 종료 ] ---


            # --- [ 수정 시작 ] --- Apply heads and return 3 values --- [ 수정 종료 ] ---
            # 갈래 1: 기존 Connector (내용 예측)
            connector_embeds = self.connector(metaquery_embeds)

            # 갈래 2: 신규 Objectness Head (유효성/길이 예측)
            # (Shape: [bsz, N_max, 1] -> [bsz, N_max]) - N_max is now max_len
            objectness_logits = self.objectness_head(metaquery_embeds).squeeze(-1)

            # Return the potentially padded attention mask
            return connector_embeds, objectness_logits, final_attention_mask
            # --- [ 수정 종료 ] ---

        # (만약 num_metaqueries == 0 이라면)
        # --- [ 수정 시작 ] --- num_metaqueries가 0일 때의 반환값 수정 --- [ 수정 종료 ] ---
        prompt_embeds = self.connector(mllm_output_embeds)
        objectness_logits = None # MetaQuery가 없으면 로ジット도 없음
        return prompt_embeds, objectness_logits, attention_mask # 전체 attention_mask 반환
        # --- [ 수정 종료 ] ---


    def forward(self, x, timestep, prompt_embeds=None, attention_mask=None):
        if isinstance(self.transformer, SanaTransformer2DModel):
            transformer_dtype = next(self.transformer.parameters()).dtype
            x = x.to(transformer_dtype)
            if prompt_embeds is not None:
                 prompt_embeds = prompt_embeds.to(transformer_dtype)
            # timestep도 float 타입으로 변환해야 할 수 있음 (Sana 모델 확인 필요)
            # timestep = timestep.to(transformer_dtype)
            model_pred = self.transformer(
                hidden_states=x,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=attention_mask, # <-- 마스크 전달
            ).sample
            return model_pred
        elif isinstance(self.transformer, UNet2DConditionModel):
            # --- [ 수정 시작 ] --- UNet도 attention_mask 전달 (cross_attention_kwargs 사용) --- [ 수정 종료 ] ---
            # UNet2DConditionModel은 encoder_attention_mask 인자를 직접 받지 않음
            # cross_attention_kwargs를 통해 전달해야 할 수 있음 (모델 구조 확인 필요)
            # 예시: cross_attention_kwargs={"attention_mask": attention_mask}
            # 정확한 사용법은 UNet 구현 확인 필요. 우선 기본 호출 유지.
            # TODO: Verify how UNet2DConditionModel uses attention_mask if needed.
            # If the UNet uses cross-attention layers that accept `encoder_attention_mask`,
            # it might need modification or passing via cross_attention_kwargs.
            # Assuming standard diffusers UNet, it might ignore the mask unless customized.
            transformer_dtype = next(self.transformer.parameters()).dtype
            x = x.to(transformer_dtype)
            if prompt_embeds is not None:
                prompt_embeds = prompt_embeds.to(transformer_dtype)
            # timestep도 float 타입으로 변환해야 할 수 있음 (UNet 모델 확인 필요)
            # timestep = timestep.to(transformer_dtype)
            model_pred = self.transformer(
                sample=x,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                # cross_attention_kwargs={"attention_mask": attention_mask} # <-- 주석 처리됨, 필요시 활성화
            ).sample
            # --- [ 수정 종료 ] ---
            return model_pred
        else:
            raise ValueError(
                f"Unsupported model: {self.transformer.__class__.__name__}"
            )