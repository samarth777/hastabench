source video/hfenv.sh
cd video/hasta
rm -rf ../snaps2 && npx -y hyperframes@latest snapshot . --at 4.6,34.5 --no-end --output ../snaps2 --timeout 20000 2>&1 | tail -2
npx -y hyperframes@latest render . --output ../hastabench_film.mp4 --quality high --fps 30 2>&1 | grep -vE "hf-progress|Render:trace" | tail -8
ls -la ../hastabench_film.mp4
