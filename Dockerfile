# Build llama.cpp quantize binary
FROM python:3.11-slim AS llama-build
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake g++ make && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /llama
RUN cmake -S /llama -B /llama/build -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    && cmake --build /llama/build --target llama-quantize -j$(nproc) \
    && mkdir -p /opt/bin && cp /llama/build/bin/llama-quantize /opt/bin/quantize

# Runtime
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY --from=llama-build /opt/bin/quantize /usr/local/bin/quantize
ENV LLAMA_QUANTIZE_BIN=/usr/local/bin/quantize
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
ENV SQ_DATA=/srv/data PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --app-dir /srv"]
