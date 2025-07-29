# Use NVIDIA Triton Inference Server with Python backend
FROM nvcr.io/nvidia/tritonserver:23.05-py3

# Install Python dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy Triton models and server code
COPY models/   /models/
COPY mytts/    /workspace/mytts/
COPY server.py /workspace/server.py

# Remove hardcoded token - will be passed at runtime
ENV HF_HUB_DISABLE_XET=1
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=${HF_HOME} 
ENV TRITON_MODEL_REPOSITORY=/models

# Expose Triton gRPC (8001) and HTTP FastAPI (8000)
EXPOSE 8001 8000 8002 8003 7004

# Create entrypoint script to handle arguments
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/workspace/entrypoint.sh"]
