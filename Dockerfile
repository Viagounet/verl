ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.6.post2
FROM ${SGLANG_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG INSTALL_ORANGE_CERTS=0
ARG EXTRA_CA_CERT_B64=""
ARG VERL_REPO=https://github.com/Viagounet/verl.git
ARG VERL_REF=main
ARG FILESDSL_REPO=https://github.com/Viagounet/FilesDSL.git
ARG FILESDSL_REF=main
ARG FLASH_ATTN_VERSION=2.8.1

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RESUME_RETRIES=20 \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_DIR=/etc/ssl/certs \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git && \
    update-ca-certificates && \
    git config --system http.sslCAInfo /etc/ssl/certs/ca-certificates.crt && \
    git config --system https.sslCAInfo /etc/ssl/certs/ca-certificates.crt && \
    pip config set global.cert /etc/ssl/certs/ca-certificates.crt && \
    rm -rf /var/lib/apt/lists/*

# Optional: install Orange corporate certificates when building inside that environment.
RUN if [ "${INSTALL_ORANGE_CERTS}" = "1" ]; then \
        curl -fsSL https://repo.yourdev.tech.orange/yourdev-install.sh | sed 's#http://#https://#g' | bash && \
        apt-get update && \
        apt-get install -y --no-install-recommends orange-certs && \
        update-ca-certificates && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Optional: inject an extra corporate CA cert via base64-encoded PEM at build time.
RUN if [ -n "${EXTRA_CA_CERT_B64}" ]; then \
        echo "${EXTRA_CA_CERT_B64}" | base64 -d > /usr/local/share/ca-certificates/extra-corp-ca.crt && \
        update-ca-certificates; \
    fi

WORKDIR /opt

RUN git clone --depth 1 --branch "${VERL_REF}" "${VERL_REPO}" /opt/verl

# The base image already contains a compatible SGLang stack.
# Install verl without [sglang] extras so core deps (e.g. ray) are present.
RUN pip install -e /opt/verl && \
    if ! python3 -c "import cachetools" >/dev/null 2>&1; then \
        pip install "cachetools>=5.3,<6"; \
    fi && \
    if ! python3 -c "import flash_attn_2_cuda" >/dev/null 2>&1; then \
        TORCH_MM="$(python3 -c 'import torch; v=torch.__version__.split("+")[0].split("."); print("{}.{}".format(v[0], v[1]))')" && \
        PY_TAG="$(python3 -c 'import sys; print("cp{}{}".format(sys.version_info.major, sys.version_info.minor))')" && \
        ABI_FLAG="$(python3 -c 'import torch; print("TRUE" if torch.compiled_with_cxx11_abi() else "FALSE")')" && \
        ARCH="$(uname -m)" && \
        case "${ARCH}" in \
            x86_64) PLAT="linux_x86_64" ;; \
            aarch64) PLAT="linux_aarch64" ;; \
            *) echo "Unsupported architecture for flash-attn wheel: ${ARCH}" && exit 1 ;; \
        esac && \
        WHEEL="flash_attn-${FLASH_ATTN_VERSION}+cu12torch${TORCH_MM}cxx11abi${ABI_FLAG}-${PY_TAG}-${PY_TAG}-${PLAT}.whl" && \
        URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${WHEEL}" && \
        pip install --no-build-isolation --no-deps "${URL}" && \
        python3 -c "import torch; import flash_attn_2_cuda"; \
    fi && \
    pip install "git+${FILESDSL_REPO}@${FILESDSL_REF}" \
        sentence-transformers \
        scikit-learn

WORKDIR /workspace

CMD ["/bin/bash"]
