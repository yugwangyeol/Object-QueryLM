# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union, List

import numpy as np
import torch
import torch.nn.functional as F # --- [ 신규 추가 ] --- F 임포트 --- [ 추가 종료 ] ---
from diffusers.models import AutoencoderKL, AutoencoderDC
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from diffusers.schedulers import (
    DDPMScheduler,
    FlowMatchEulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import PreTrainedModel
import PIL
from tqdm import tqdm

from models.model import MLLMInContextConfig, MLLMInContext
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)


class MetaQueryConfig(MLLMInContextConfig):
    model_type = "metaquery"

    def __init__(
        self,
        vae_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        input_size: int = 16,
        in_channels: int = 32,
        vae_downsample_f: int = 32,
        noise_scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        _gradient_checkpointing: bool = True,
        loss_type: str = "flow",
        num_metaqueries: int = 64,
        modules_to_freeze: tuple[str] = (),
        modules_to_unfreeze: tuple[str] = (),
        # --- [ 신규 추가 ] --- Loss 가중치 설정 추가 --- [ 추가 종료 ] ---
        lambda_length: float = 0.1, # Length Loss 가중치
        # --- [ 신규 추가 ] --- 추론 임계값 설정 추가 --- [ 추가 종료 ] ---
        inference_threshold: float = 0.5, # 추론 시 사용할 확률 임계값
        **kwargs,
    ):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.vae_id = vae_id
        self.input_size = input_size
        self.in_channels = in_channels
        self.vae_downsample_f = vae_downsample_f
        self.noise_scheduler_id = noise_scheduler_id
        self.scheduler_id = scheduler_id
        self._gradient_checkpointing = _gradient_checkpointing
        self.loss_type = loss_type
        self.num_metaqueries = num_metaqueries
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_unfreeze = modules_to_unfreeze
        # --- [ 신규 추가 ] --- 설정 값 저장 --- [ 추가 종료 ] ---
        self.lambda_length = lambda_length
        self.inference_threshold = inference_threshold


class MetaQuery(PreTrainedModel):
    config_class = MetaQueryConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

        self.model = MLLMInContext(MLLMInContextConfig(**config.to_dict()))
        self.loss_type = config.loss_type

        if "Sana" in config.vae_id:
            self.vae = AutoencoderDC.from_pretrained(config.vae_id, subfolder="vae")
        else:
            try:
                self.vae = AutoencoderKL.from_pretrained(config.vae_id)
            except:
                self.vae = AutoencoderKL.from_pretrained(config.vae_id, subfolder="vae")

        if self.loss_type == "flow":
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                config.noise_scheduler_id, subfolder="scheduler"
            )
        elif self.loss_type == "diff":
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                config.noise_scheduler_id, subfolder="scheduler"
            )
        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")

        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            config.scheduler_id, subfolder="scheduler"
        )

        for module_name in config.modules_to_freeze:
            if "." in module_name:
                module = self
                for sub_module_name in module_name.split("."):
                    module = getattr(module, sub_module_name, None)
                    if module is None:
                        break
                else:
                    module.requires_grad_(False)
            else:
                module = getattr(self, module_name, None)
                if module is not None:
                    module.requires_grad_(False)

        for module_name in config.modules_to_unfreeze:
            if "." in module_name:
                module = self
                for sub_module_name in module_name.split("."):
                    module = getattr(module, sub_module_name, None)
                    if module is None:
                        break
                else:
                    module.requires_grad_(True)
            else:
                module = getattr(self, module_name, None)
                if module is not None:
                    module.requires_grad_(True)

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_tokenizer(self):
        return self.model.get_tokenizer()

    def get_tokenize_fn(self):
        return self.model.get_tokenize_fn()

    # --- [ 수정 시작: forward 함수 시그니처 변경 및 로직 수정 ] ---
    def forward(
        self,
        target,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        object_count_k=None, # <-- [ 1. 인자 추가: dataset.py에서 'k'값을 받음 ]
        **kwargs
    ):
    # --- [ 수정 종료 ] ---
        if self.vae is not None:
            if isinstance(self.vae, AutoencoderKL):
                latents = self.vae.encode(target).latent_dist.sample()
            elif isinstance(self.vae, AutoencoderDC):
                latents = self.vae.encode(target).latent
            else:
                raise ValueError(f"Unknown vae type {type(self.vae)}")
            if (
                hasattr(self.vae.config, "shift_factor") # --- [ 수정 시작 ] --- Check attribute existence --- [ 수정 종료 ] ---
                and self.vae.config.shift_factor is not None
            ):
                latents = latents - self.vae.config.shift_factor
            latents = latents * self.vae.config.scaling_factor
        else:
            latents = target

        bsz = latents.shape[0]

        # QwenVL pixel_values 처리 (기존 코드 유지)
        if (
            pixel_values is not None
            and hasattr(self.model, "mllm_type")
            and self.model.mllm_type == "qwenvl"
        ):
            # QwenVL expects pixel_values to be a single tensor, not nested list
            # We assume tokenize function handles the conversion to the correct format if needed
            # If pixel_values is already shape [bsz*num_images, c, h, w], no need to squeeze(0)
            # Check shape, if [1, bsz*num_images, c, h, w], then squeeze(0)
            if pixel_values.ndim == 5 and pixel_values.shape[0] == 1:
                 pixel_values = pixel_values.squeeze(0)

        noise = torch.randn_like(latents, device=latents.device)

        # --- [ 수정 시작: 핵심 로직 분리 및 loss_length 계산 ] ---
        # Initialize loss_length and final_diffusion_attention_mask
        loss_length = torch.tensor(0.0, device=latents.device, dtype=target.dtype) # dtype 일치
        loss_recon = torch.tensor(0.0, device=latents.device, dtype=target.dtype) # dtype 일치

        # [ 2. MLLM에서 3가지 값 받아오기 ]
        prompt_embeds, objectness_logits, base_attention_mask = self.model.encode_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=kwargs.get("image_sizes", None),
        )

        final_diffusion_attention_mask = base_attention_mask # 기본 마스크로 초기화

        # [ 4. 핵심 로직: k값이 제공된 경우 (학습 시) ]
        if objectness_logits is not None and object_count_k is not None:
            # Check if mask dimensions match logits dimensions after padding in encode_condition
            N_max_logits = objectness_logits.shape[1]
            N_max_mask = base_attention_mask.shape[1]
            # Ensure N_max matches between logits and mask (due to potential padding)
            if N_max_logits != N_max_mask:
                 # This might happen if tokenization/padding logic changes.
                 # Adjust smaller dimension or handle error. For now, assume they match or truncate.
                 print(f"Warning: Mismatch N_max between logits ({N_max_logits}) and mask ({N_max_mask}). Truncating.")
                 min_n_max = min(N_max_logits, N_max_mask)
                 objectness_logits = objectness_logits[:, :min_n_max]
                 base_attention_mask = base_attention_mask[:, :min_n_max]
                 N_max = min_n_max
            else:
                 N_max = N_max_logits

            device = objectness_logits.device

            # [ 4A. 정답 마스크 생성 (k = n*j) ]
            arange_tensor = torch.arange(N_max, device=device).unsqueeze(0)
            # object_count_k는 이미 n*j 값임, Clamp k to be <= N_max
            clamped_k = torch.clamp(object_count_k, max=N_max)
            target_mask = (arange_tensor < clamped_k.unsqueeze(-1)).float()

            # [ 4B. Length Loss 계산 (BCE) ]
            loss_length = F.binary_cross_entropy_with_logits(
                objectness_logits.float(), target_mask.float(), reduction="mean" # Use float() for stability
            )

            # [ 4C. Diffusion 마스크 생성 (Teacher Forcing) ]
            # 학습 안정성을 위해 MLLM의 예측이 아닌 '정답' 마스크를 사용
            # base_attention_mask는 패딩 마스크 역할
            final_diffusion_attention_mask = base_attention_mask * target_mask
        # --- [ 수정 종료 ] ---


        if self.loss_type == "flow":
            # Flow Loss 계산 로직 (기존 코드 기반 + 마스크 적용)
            weighting_scheme = "uniform" # or other scheme
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=bsz,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(
                device=latents.device
            )

            sigmas = self.get_sigmas(
                timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype
            )
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

            # [ 5. 마스킹된 Diffusion 호출 ]
            model_pred = self.model(
                x=noisy_latents,
                timestep=timesteps,
                prompt_embeds=prompt_embeds,
                attention_mask=final_diffusion_attention_mask, # <-- 수정된 마스크 사용
            )

            target_flow = noise - latents # Flow matching target
            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=weighting_scheme, sigmas=sigmas
            )
            # Calculate reconstruction loss (flow matching loss)
            loss_recon = torch.mean(
                (
                    weighting.float() * (model_pred.float() - target_flow.float()) ** 2
                ).reshape(target_flow.shape[0], -1),
                1,
            )
            loss_recon = loss_recon.mean()


        elif self.loss_type == "diff":
            # Diffusion Loss 계산 로직 (기존 코드 기반 + 마스크 적용)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            if self.noise_scheduler.config.prediction_type == "epsilon":
                target_diff = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target_diff = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            # [ 5. 마스킹된 Diffusion 호출 ]
            noise_pred = self.model(
                x=noisy_latents,
                timestep=timesteps,
                prompt_embeds=prompt_embeds,
                attention_mask=final_diffusion_attention_mask, # <-- 수정된 마스크 사용
            )
            # Calculate reconstruction loss (diffusion loss, usually MSE)
            loss_recon = F.mse_loss(noise_pred.float(), target_diff.float(), reduction="mean")

        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")

        # --- [ 6. 최종 Loss 반환 (수정) ] ---
        # lambda_length는 config에서 읽어옴
        lambda_length = self.config.lambda_length
        loss = loss_recon + lambda_length * loss_length

        return {"loss": loss, "loss_recon": loss_recon, "loss_length": loss_length}
        # --- [ 수정 종료 ] ---

    @torch.no_grad()
    def decode_latents(self, latents, normalize=True, return_tensor=False):
        if self.vae is not None:
            latents = latents / self.vae.config.scaling_factor
            if (
                hasattr(self.vae.config, "shift_factor") # --- [ 수정 시작 ] --- Check attribute existence --- [ 수정 종료 ] ---
                and self.vae.config.shift_factor is not None
            ):
                latents = latents + self.vae.config.shift_factor
            samples = self.vae.decode(latents).sample
        else:
            samples = latents
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)
        if return_tensor:
            return samples
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples

    # --- [ 수정 시작: sample_images 함수 로직 수정 ] ---
    @torch.no_grad() # @torch.no_grad() decorator added
    def sample_images(
        self,
        caption="",
        input_images=None, # Accepts single PIL Image or list of PIL Images for our case
        guidance_scale: float = 3.0,
        image_guidance_scale: float = 1.5, # Keep for potential future use, but not used in img-only
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1, # Usually 1 for reconstruction/validation
        return_tensor=False,
        negative_prompt="", # Keep for potential future use
        enable_progress_bar=False,
        **kwargs,
    ):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype # Get model dtype

        # Handle caption input (ensure it's a list)
        if not isinstance(caption, list):
            caption = [caption] * (len(input_images) if input_images is not None else 1) # Match batch size if images provided
        bsz = len(caption)

        # Handle image input (ensure nested list format [batch_size, num_input_images_per_item])
        if input_images is not None:
            if not isinstance(input_images, list): # Single image case
                input_images = [[input_images]] * bsz # Repeat for batch size
            elif not isinstance(input_images[0], list): # List of single images case
                 input_images = [[img] for img in input_images]
            # Ensure outer list length matches batch size
            if len(input_images) != bsz:
                 # If single list provided for multiple captions, repeat it
                 if len(input_images) == 1 and bsz > 1:
                      input_images = input_images * bsz
                 else: # Or adjust batch size to match images
                      bsz = len(input_images)
                      caption = caption[:bsz] # Truncate captions if needed
        else:
             # If no image provided, we need different logic or raise error for img-only model
             # For now, let's assume image is always provided for this mode.
             if not any(caption): # If caption is also empty, nothing to condition on
                  raise ValueError("Either caption or input_images must be provided.")
             # If only caption, proceed as text-to-image (but our focus is img-only)


        # Classifier-Free Guidance (CFG) setup
        # For image-only, CFG might mean guidance between image condition and unconditional.
        # Original code uses text CFG. We'll adapt for image CFG if image_guidance_scale > 1.0
        # For simplicity in img-only reconstruction, we might set guidance_scale=1.0 (no CFG) initially.
        # Let's keep the structure but note its adaptation.

        do_text_guidance = guidance_scale > 1.0
        # Image guidance needs adaptation: unconditional input would be a null image.
        # Let's disable image CFG for now in this image-only setting unless explicitly needed.
        do_image_guidance = False # Assume image_guidance_scale is not used for simple reconstruction
        if input_images is None:
             do_image_guidance = False # Can't do image guidance without images

        tokenize_func = self.get_tokenize_fn()
        tokenizer = self.get_tokenizer()

        # Prepare inputs for MLLM encoding
        # We need conditional inputs (image + empty caption)
        # And potentially unconditional inputs (null image + empty caption or just empty caption)

        cond_captions = caption
        cond_images = input_images

        uncond_captions = [negative_prompt] * bsz
        # For unconditional, we might use a null image or just rely on text
        # If focusing purely on reconstruction, unconditional part might be skipped (guidance_scale=1)
        # Let's create null images for potential unconditional image input
        uncond_images = None
        if input_images:
            uncond_images = [
                 [PIL.Image.new("RGB", img.size, (127, 127, 127)) for img in img_list] if img_list else None # Use grey image
                 for img_list in input_images
            ]

        # Combine inputs for batch processing by tokenize func
        combined_captions = []
        combined_images = []

        if do_text_guidance:
            combined_captions.extend(uncond_captions)
            combined_images.extend(uncond_images if uncond_images is not None else [None]*bsz) # Use null images if available

        combined_captions.extend(cond_captions)
        combined_images.extend(cond_images if cond_images is not None else [None]*bsz) # Use real images

        # Tokenize all inputs together
        tokenized_outputs = tokenize_func(
            tokenizer, combined_captions, combined_images
        )
        input_ids, attention_mask, pixel_values, image_sizes = tokenized_outputs

        # Encode conditions using MLLM
        prompt_embeds, objectness_logits, base_attention_mask = self.model.encode_condition(
            input_ids=input_ids.to(device=device, dtype=torch.long),
            attention_mask=attention_mask.to(device=device, dtype=torch.long),
            pixel_values=pixel_values.to(device=device, dtype=dtype) if pixel_values is not None else None,
            image_sizes=image_sizes.to(device=device, dtype=torch.long) if image_sizes is not None else None,
        )

        # [ 2. 추론용 동적 마스크 생성 ]
        final_diffusion_attention_mask = base_attention_mask # Start with the base mask (handles padding)

        if objectness_logits is not None:
            # We only care about the logits from the conditional part
            if do_text_guidance:
                 cond_logits = objectness_logits[bsz:] # Get logits for the conditional inputs
            else:
                 cond_logits = objectness_logits

            # MLLM의 예측 확률 (logits -> 0~1)
            objectness_probs = torch.sigmoid(cond_logits.float()) # Use float for stability

            # config에서 임계값 읽어오기
            threshold = self.config.inference_threshold
            inference_mask = (objectness_probs > threshold).float()

            # 최종 마스크 = MLLM 예측 * 기존 패딩 마스크
            # Make sure base_attention_mask corresponds to the conditional part
            if do_text_guidance:
                 cond_base_mask = base_attention_mask[bsz:]
            else:
                 cond_base_mask = base_attention_mask

            # Ensure shapes match before multiplication (due to potential padding)
            mask_len = inference_mask.shape[1]
            if cond_base_mask.shape[1] != mask_len:
                 print(f"Warning: Mismatch length in inference mask generation. Truncating base mask.")
                 cond_base_mask = cond_base_mask[:, :mask_len]

            final_diffusion_attention_mask = cond_base_mask * inference_mask

            # If using CFG, we need the mask for unconditional part too (usually all ones or all zeros?)
            # For simplicity, let's assume uncond uses the base mask without objectness filtering,
            # or uses a mask of all zeros if we want zero conditioning. Let's use the base mask.
            if do_text_guidance:
                 uncond_final_mask = base_attention_mask[:bsz]
                 # Pad unconditional mask if needed to match conditional mask length
                 if uncond_final_mask.shape[1] < final_diffusion_attention_mask.shape[1]:
                      pad_len = final_diffusion_attention_mask.shape[1] - uncond_final_mask.shape[1]
                      uncond_final_mask = F.pad(uncond_final_mask, (0, pad_len), value=0)
                 elif uncond_final_mask.shape[1] > final_diffusion_attention_mask.shape[1]:
                      uncond_final_mask = uncond_final_mask[:, :final_diffusion_attention_mask.shape[1]]

                 final_diffusion_attention_mask = torch.cat([uncond_final_mask, final_diffusion_attention_mask], dim=0)

        # Handle num_images_per_prompt > 1 (repeat embeds and masks)
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            if final_diffusion_attention_mask is not None:
                final_diffusion_attention_mask = final_diffusion_attention_mask.repeat_interleave(num_images_per_prompt, dim=0)


        # Prepare latent variables
        latent_size = self.config.input_size
        latent_channels = self.config.in_channels
        latents = randn_tensor(
            shape=(
                bsz * num_images_per_prompt, # Adjust shape for num_images_per_prompt
                latent_channels,
                latent_size,
                latent_size,
            ),
            generator=generator,
            device=device,
            dtype=dtype, # Use model dtype
        )
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Diffusion loop
        for t in tqdm(
            timesteps,
            desc="Sampling images",
            disable=not enable_progress_bar,
        ):
            # Expand the latents if we are doing classifier free guidance
            # Input to diffusion model needs potentially duplicated latents if CFG is on
            latent_model_input = torch.cat([latents] * 2) if do_text_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.to(dtype) # Ensure correct dtype

            # Predict the noise residual
            noise_pred = self.model(
                x=latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]), # Expand timestep
                prompt_embeds=prompt_embeds, # Already includes uncond and cond if CFG
                attention_mask=final_diffusion_attention_mask, # Use the final combined mask
            )

            # Perform guidance
            if do_text_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            # else: noise_pred remains as is

            # Compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        # Ensure latents are scaled correctly before decoding if VAE expects it
        samples = self.decode_latents(
            latents.to(self.vae.dtype) if self.vae is not None else latents, # Use VAE dtype
            return_tensor=return_tensor,
        )
        return samples
    # --- [ 수정 종료 ] ---