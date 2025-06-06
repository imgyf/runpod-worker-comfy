# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
  python3.10 \
  python3-pip \
  git \
  wget \
  libgl1 \
  && ln -sf /usr/bin/python3.10 /usr/bin/python \
  && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install comfy-cli
RUN pip install comfy-cli

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 11.8 --nvidia --version 0.3.26

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install runpod
RUN pip install runpod requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add scripts
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh /restore_snapshot.sh

# Optionally copy the snapshot file
ADD *snapshot*.json /

# Restore the snapshot to install custom nodes
RUN /restore_snapshot.sh

# Start container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base as downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories
RUN mkdir -p models/checkpoints models/vae models/clip models/upscale_models models/florence2/Florence-2-Flux-Large models/loras

# Download all models
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/flux1-dev-fp8.safetensors https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/flux1-redux-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.sft https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/clip/sigclip_vision_patch14_384.safetensors https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/upscale_models/4x-ClearRealityV1.pth https://huggingface.co/skbhadra/ClearRealityV1/resolve/main/4x-ClearRealityV1.pth && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/florence2/Florence-2-Flux-Large/model.safetensors https://huggingface.co/gokaygokay/Florence-2-Flux-Large/resolve/main/model.safetensors && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/florence2/Florence-2-Flux-Large/config.json https://huggingface.co/gokaygokay/Florence-2-Flux-Large/resolve/main/config.json && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/florence2/Florence-2-Flux-Large/configuration_florence2.py https://huggingface.co/gokaygokay/Florence-2-Flux-Large/resolve/main/configuration_florence2.py && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/florence2/Florence-2-Flux-Large/generation_config.json https://huggingface.co/gokaygokay/Florence-2-Flux-Large/resolve/main/generation_config.json && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/florence2/Florence-2-Flux-Large/modeling_florence2.py https://huggingface.co/gokaygokay/Florence-2-Flux-Large/resolve/main/modeling_florence2.py && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/florence2/Florence-2-Flux-Large/preprocessor_config.json https://huggingface.co/gokaygokay/Florence-2-Flux-Large/resolve/main/preprocessor_config.json && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/florence2/Florence-2-Flux-Large/processing_florence2.py https://huggingface.co/gokaygokay/Florence-2-Flux-Large/resolve/main/processing_florence2.py && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/florence2/Florence-2-Flux-Large/tokenizer.json https://huggingface.co/gokaygokay/Florence-2-Flux-Large/resolve/main/tokenizer.json && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/florence2/Florence-2-Flux-Large/tokenizer_config.json https://huggingface.co/gokaygokay/Florence-2-Flux-Large/resolve/main/tokenizer_config.json && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/florence2/Florence-2-Flux-Large/vocab.json https://huggingface.co/gokaygokay/Florence-2-Flux-Large/resolve/main/vocab.json && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/florence2/Florence-2-Flux-Large/added_tokens.json https://huggingface.co/gokaygokay/Florence-2-Flux-Large/resolve/main/added_tokens.json && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/florence2/Florence-2-Flux-Large/special_tokens_map.json https://huggingface.co/gokaygokay/Florence-2-Flux-Large/resolve/main/special_tokens_map.json && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/florence2/Florence-2-Flux-Large/merges.txt https://huggingface.co/gokaygokay/Florence-2-Flux-Large/resolve/main/merges.txt && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/ins_style.safetensors https://huggingface.co/imgyf/nexera-models/resolve/main/ins_style.safetensors && \
  wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/loras/ins_lisa.safetensors https://huggingface.co/imgyf/nexera-models/resolve/main/ins_lisa.safetensors

# Stage 3: Final image
FROM base as final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Start container
CMD ["/start.sh"]
