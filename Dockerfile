FROM rocm/pytorch:latest

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN /opt/conda/bin/python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x run.sh

ENTRYPOINT ["./run.sh"]