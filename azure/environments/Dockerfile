FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:20250828.v1

SHELL ["/bin/bash", "-c"]

COPY conda_env.yml /tmp/conda_env.yml

# Install all runtime dependencies into the base conda environment once,
# so AML jobs can pull the prebuilt image instead of rebuilding an env.
RUN conda env update -n base -f /tmp/conda_env.yml && \
    conda clean -a -y && \
    rm -f /tmp/conda_env.yml

ENV PATH=/opt/miniconda/bin:$PATH
