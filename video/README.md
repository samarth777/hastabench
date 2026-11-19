# HastaBench film

`hastabench_film.mp4`: a 66 s, 1920×1080, 30 fps overview of the benchmark, built with [HyperFrames](https://github.com/heygen-com/hyperframes).

* `hasta/index.html` and `hasta/film.js` hold the composition: a GSAP timeline plus two deterministic three.js frame sources that skin the benchmark's hand mesh from logged joint angles.
* `prep_video_data.py` regenerates `hasta/assets/` (data and stills) from the report data. Run it from the repo root as a GPU job: `scripts/cluster/job.sh video/prep_video_data.py`.
* `render.sh` renders the film; `hfenv.sh` installs Node 22 and a static FFmpeg into node-local /tmp. Usage: `bash video/render.sh`, from the repo root.
