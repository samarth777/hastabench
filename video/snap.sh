source video/hfenv.sh
cd video/hasta
npx -y hyperframes@latest lint . 2>&1 | grep -vE "nested_structure|Move <|Fix: Move" | tail -12
rm -rf ../snaps && npx -y hyperframes@latest snapshot . --at 1.0,4.6,7.0,12.0,17.0,24.5,27.0,34.5,45.0,48.5,55.0,62.0 --no-end --output ../snaps --timeout 20000 2>&1 | tail -6
