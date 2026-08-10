# Reproducibility image for AutoTuneTurboQuant.
# The official vLLM image pins CUDA, PyTorch, and vLLM together — we add
# only the analysis/profiler dependencies on top.
#
# Note: many SLURM clusters run containers via Apptainer/Singularity:
#   apptainer build autotune-tq.sif docker://<this image after build+push>
# A plain venv from requirements.txt is an equally supported path; this
# image exists so the grader can reproduce without touching the host env.
FROM vllm/vllm-openai:v0.21.0

WORKDIR /workspace/autotune-turboquant
COPY requirements.txt .
# vLLM itself ships with the base image; install everything else.
RUN grep -v "^vllm==" requirements.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt

COPY . .
ENTRYPOINT []
CMD ["/bin/bash"]
