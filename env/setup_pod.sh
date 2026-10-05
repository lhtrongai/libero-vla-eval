#!/usr/bin/env bash
# Run once per new pod: system packages, LIBERO-Plus assets on local disk, terminal look.
set -e
mkdir -p ~/.jupyter/lab && cp -r /workspace/.jupyter-settings ~/.jupyter/lab/user-settings
grep -q prompt.sh ~/.bashrc || echo 'source /workspace/.prompt.sh' >> ~/.bashrc
bash /workspace/libero-vla-eval/env/setup_env.sh
apt-get install -y -qq libmagickwand-dev > /dev/null 2>&1 && echo "[ok] ImageMagick"
pip install -q Wand "scikit-image<0.25" numpy==1.26.4 && echo "[ok] Wand, scikit-image"
if [ ! -d /opt/libero-plus-assets ]; then
  unzip -q /workspace/libero-plus-assets.zip -d /opt/lp_unzip
  mv /opt/lp_unzip/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets /opt/libero-plus-assets
  rm -rf /opt/lp_unzip
fi
ls /workspace/LIBERO-plus/libero/libero/assets/scenes > /dev/null && echo "[ok] LIBERO-Plus assets"
