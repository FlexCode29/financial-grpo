FROM rocm/vllm:rocm6.3.1_vllm_0.8.5_20250513

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

#RUN /opt/conda/bin/python -m pip install --no-cache-dir --pre torch torchvision  --index-url https://download.pytorch.org/whl/nightly/rocm6.2/


RUN git clone https://github.com/billishyahao/unsloth.git && \
    cd unsloth && \
    git checkout billhe/rocm && \
    pip install --no-cache-dir . && \
    cd .. && rm -rf unsloth

RUN pip install --no-cache-dir unsloth_zoo==2025.3.17

# -----------------------------------------------------------------------------
# Build and install ROCm‑enabled bitsandbytes (required by Unsloth)
# -----------------------------------------------------------------------------
RUN git clone --recurse-submodules https://github.com/ROCm/bitsandbytes.git && \
    cd bitsandbytes && \
    git checkout rocm_enabled_multi_backend && \
    pip install --no-cache-dir -r requirements-dev.txt && \
    cmake -DCOMPUTE_BACKEND=hip -S . && \
    make -j$(nproc) && \
    pip install --no-cache-dir . && \
    cd .. && rm -rf bitsandbytes

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install deepspeed==0.16.4

RUN pip install trl==0.21.0  --no-dependencies

COPY . .

RUN chmod +x run.sh

ENTRYPOINT ["./run.sh"]