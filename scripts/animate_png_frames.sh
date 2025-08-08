#!/bin/bash
set -e

# Check if directory argument is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/image_directory"
  exit 1
fi

IMG_DIR="$1"

# Check if directory exists
if [ ! -d "$IMG_DIR" ]; then
  echo "Error: Directory '$IMG_DIR' not found."
  exit 1
fi

cd "$IMG_DIR"

# Find all unique patterns like x-BField_xy, y-BField_xz, etc.
patterns=$(ls *_plt*_slice.png 2>/dev/null \
  | sed -E 's/.*_plt[0-9]+_([a-z]-BField_[a-z]{2}).*/\1/' \
  | sort -u)

if [ -z "$patterns" ]; then
  echo "No matching PNG files found in $IMG_DIR"
  exit 1
fi

FRAMERATE=10

for pat in $patterns; do
  echo "Processing pattern: $pat"

  ffmpeg -framerate $FRAMERATE -pattern_type glob \
    -i "*_plt*_${pat}_slice.png" \
    -c:v libx264 -pix_fmt yuv420p \
    "${pat}.mp4"
done

echo "All videos are saved in $IMG_DIR"


## .