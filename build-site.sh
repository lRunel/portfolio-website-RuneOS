#!/usr/bin/env bash
# Copies only the files the website needs into dist/ (the folder to deploy).
# Rebuild the kernel first with `make -C kernel` if you changed any C code.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf dist
mkdir -p dist
cp index.html dist/
cp -r css js wasm projects dist/
mkdir -p dist/assets
cp -r assets/wallpaper assets/favicon.jpg assets/Shubham_Chowta_SWE_Backend.pdf dist/assets/
echo "dist/ ready ($(du -sh dist | cut -f1)) — deploy with: npx wrangler pages deploy dist"
