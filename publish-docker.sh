#!/bin/bash
# Publish PersonaPlex RunPod container to Docker Hub
#
# Usage: ./publish-docker.sh <dockerhub-username> [tag]
# Example: ./publish-docker.sh myusername v1.0.0

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <dockerhub-username> [tag]"
    echo "Example: $0 myusername v1.0.0"
    exit 1
fi

DOCKERHUB_USER="$1"
TAG="${2:-latest}"
IMAGE_NAME="personaplex-runpod"
FULL_TAG="${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"

echo "─────────────────────────────────────────────────────────"
echo "  Publishing PersonaPlex RunPod Container"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "  Local image: ${IMAGE_NAME}:latest"
echo "  Docker Hub:  ${FULL_TAG}"
echo ""

# Check if local image exists
if ! docker image inspect "${IMAGE_NAME}:latest" >/dev/null 2>&1; then
    echo "❌ Error: Local image '${IMAGE_NAME}:latest' not found"
    echo "   Build it first with: docker build -f Dockerfile.personaplex -t ${IMAGE_NAME}:latest ."
    exit 1
fi

# Tag the image
echo "📦 Tagging image..."
docker tag "${IMAGE_NAME}:latest" "${FULL_TAG}"

# Check if logged in
if ! docker info | grep -q "Username:"; then
    echo ""
    echo "⚠️  Not logged into Docker Hub"
    echo "   Log in with: docker login"
    echo ""
    read -p "Press Enter to continue (or Ctrl+C to cancel)..."
fi

# Push to Docker Hub
echo ""
echo "🚀 Pushing to Docker Hub..."
docker push "${FULL_TAG}"

echo ""
echo "✅ Successfully published: ${FULL_TAG}"
echo ""
echo "  Use this image in RunPod Serverless:"
echo "    ${FULL_TAG}"
echo ""
