#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 generate-icon.py
sips -s format png AppIcon.ppm --out AppIcon-1024.png >/dev/null
rm -rf AppIcon.iconset
mkdir AppIcon.iconset
sips -z 16 16 AppIcon-1024.png --out AppIcon.iconset/icon_16x16.png >/dev/null
sips -z 32 32 AppIcon-1024.png --out AppIcon.iconset/icon_16x16@2x.png >/dev/null
sips -z 32 32 AppIcon-1024.png --out AppIcon.iconset/icon_32x32.png >/dev/null
sips -z 64 64 AppIcon-1024.png --out AppIcon.iconset/icon_32x32@2x.png >/dev/null
sips -z 128 128 AppIcon-1024.png --out AppIcon.iconset/icon_128x128.png >/dev/null
sips -z 256 256 AppIcon-1024.png --out AppIcon.iconset/icon_128x128@2x.png >/dev/null
sips -z 256 256 AppIcon-1024.png --out AppIcon.iconset/icon_256x256.png >/dev/null
sips -z 512 512 AppIcon-1024.png --out AppIcon.iconset/icon_256x256@2x.png >/dev/null
sips -z 512 512 AppIcon-1024.png --out AppIcon.iconset/icon_512x512.png >/dev/null
sips -z 1024 1024 AppIcon-1024.png --out AppIcon.iconset/icon_512x512@2x.png >/dev/null
iconutil -c icns AppIcon.iconset -o AppIcon.icns
rm -rf AppIcon.iconset AppIcon.ppm
printf '%s\n' "$(pwd)/AppIcon.icns"
