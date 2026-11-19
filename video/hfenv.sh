# Source me inside an LSF job: installs Node 22 + static FFmpeg into node-local /tmp (cached per node).
export HF_TMP=/tmp/hf_$USER
mkdir -p $HF_TMP && cd $HF_TMP
if [ ! -x $HF_TMP/node/bin/node ]; then
  NV=$(curl -s https://nodejs.org/dist/index.json | python3 -c "import sys,json;print([r['version'] for r in json.load(sys.stdin) if r['version'].startswith('v22.')][0])")
  curl -sL https://nodejs.org/dist/$NV/node-$NV-linux-x64.tar.xz | tar xJ && mv node-$NV-linux-x64 node
fi
if [ ! -x $HF_TMP/ffmpeg/ffmpeg ]; then
  curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar xJ && mv ffmpeg-*-amd64-static ffmpeg
fi
export PATH=$HF_TMP/node/bin:$HF_TMP/ffmpeg:$PATH
export npm_config_cache=$HF_TMP/npm HYPERFRAMES_CACHE_DIR=$HF_TMP/hfcache HYPERFRAMES_NO_UPDATE_CHECK=1 HYPERFRAMES_SKIP_SKILLS=1 PUPPETEER_CACHE_DIR=$HF_TMP/puppeteer
cd - >/dev/null
