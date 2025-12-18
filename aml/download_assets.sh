#!/bin/bash

# Script to download assets from GitHub releases
# This script downloads required assets for leisaac

set -e  # Exit on error

echo "Starting asset download..."

# Base URL for releases
BASE_URL="https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0"

# Get the script directory and calculate project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create destination directories if they don't exist
mkdir -p "$PROJECT_ROOT/assets/scenes"
mkdir -p "$PROJECT_ROOT/assets/robots"

echo "Downloading kitchen_with_orange.zip..."
curl -L -o "$PROJECT_ROOT/kitchen_with_orange.zip" "${BASE_URL}/kitchen_with_orange.zip"

echo "Extracting kitchen_with_orange.zip to $PROJECT_ROOT/assets/scenes..."
unzip -o "$PROJECT_ROOT/kitchen_with_orange.zip" -d "$PROJECT_ROOT/assets/scenes"

echo "Removing kitchen_with_orange.zip..."
rm "$PROJECT_ROOT/kitchen_with_orange.zip"

echo "Downloading so101_follower.usd..."
curl -L -o "$PROJECT_ROOT/assets/robots/so101_follower.usd" "${BASE_URL}/so101_follower.usd"

echo "Asset download complete!"
echo "Assets have been downloaded to:"
echo "  - assets/scenes/ (kitchen_with_orange)"
echo "  - assets/robots/so101_follower.usd"
