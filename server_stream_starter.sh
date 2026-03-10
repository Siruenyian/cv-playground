mkdir -p /app/mtx && cd /app/mtx
if [ ! -f "/app/mtx/mediamtx" ]; then
    wget https://github.com/bluenviron/mediamtx/releases/download/v1.9.0/mediamtx_v1.9.0_linux_amd64.tar.gz
    tar -xzf mediamtx_v1.9.0_linux_amd64.tar.gz
fi
./mediamtx & python /app/scripts/yolox_sg_stream.py