#!/bin/bash

# Script to download assets from GitHub releases
# This script downloads required assets for leisaac

set -e  # Exit on error

echo "Starting asset download..."

# Base URL for releases
BASE_URL="https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0"

# Create destination directories if they don't exist (relative to repository root)
mkdir -p ../assets/scenes
mkdir -p ../assets/robots

echo "Downloading kitchen_with_orange.zip..."
curl -L -o kitchen_with_orange.zip "${BASE_URL}/kitchen_with_orange.zip"

echo "Extracting kitchen_with_orange.zip to ../assets/scenes..."
unzip -o kitchen_with_orange.zip -d ../assets/scenes

echo "Removing kitchen_with_orange.zip..."
rm kitchen_with_orange.zip

echo "Downloading so101_follower.usd..."
curl -L -o ../assets/robots/so101_follower.usd "${BASE_URL}/so101_follower.usd"

echo "Asset download complete!"
echo "Assets have been downloaded to:"
echo "  - ../assets/scenes/ (kitchen_with_orange)"
echo "  - ../assets/robots/so101_follower.usd"
