# Linux environment for the bash experiment launchers.
# Dependencies match uv.lock (including the CUDA PyTorch wheel). The source
# tree is bind-mounted at run time, so this image does not copy the repo.

FROM python:3.12-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates libgomp1 make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lifemanifold
COPY pyproject.toml uv.lock ./

ENV UV_PROJECT_ENVIRONMENT=/opt/lifemanifold/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON=/usr/local/bin/python3

RUN uv sync --frozen --all-groups --python /usr/local/bin/python3

COPY docker/entrypoint.sh /usr/local/bin/lifemanifold-entrypoint
RUN sed -i 's/\r$//' /usr/local/bin/lifemanifold-entrypoint \
    && chmod +x /usr/local/bin/lifemanifold-entrypoint

WORKDIR /workspace
ENTRYPOINT ["lifemanifold-entrypoint"]
CMD ["bash"]
