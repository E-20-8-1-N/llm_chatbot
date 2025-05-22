FROM ethanhph/langchain:dev

# Environment variables from your original Dockerfile (can also be set in TrueNAS YAML)
ENV DEBIAN_FRONTEND=noninteractive \
    ROCM_PATH=/opt/rocm-6.4.0 \
    HIPDIR=/opt/rocm-6.4.0 \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    HCC_AMDGPU_TARGET=gfx1030 \
    LD_LIBRARY_PATH=$ROCM_PATH/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
    MIOPEN_FIND_MODE=1 \
    HIP_VISIBLE_DEVICES=0 \
    HIP_PLATFORM=amd \
    OPENBLASDIR=/usr/lib/x86_64-linux-gnu \
    # PYTHONUNBUFFERED is good for Docker logs
    PYTHONUNBUFFERED=1 \
    # Django specific
    DJANGO_SETTINGS_MODULE=chatbot_project.settings

# Set working directory
WORKDIR /app

# Copy your Django application code into the image
COPY . /app/

# (Optional) Install/update any additional Python packages if needed
# RUN pip install --no-cache-dir -r requirements.txt

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Expose the port Gunicorn will run on
EXPOSE 8000

# Run the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]