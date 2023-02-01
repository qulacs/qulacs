#!/bin/bash

set -eu

usage() {
    echo "Usage: ./download_wheel.sh"
    echo ""
    echo "Download an artifact published from the current branch and unzip it to ./wheels/"
    echo "Required tools: gh(GitHub CLI), jq and git"
}

if ! command -v gh &> /dev/null ; then
    echo "Error: gh not found"
    usage
    exit 1
fi

# Filter URLs of each archive which is published from the current branch and retrieve the newest one.
ARTIFACT_URL=$(gh api -H "Accept: application/vnd.github+json" /repos/qulacs/qulacs/actions/artifacts \
    --jq ".artifacts[] | select(.name == \"artifact\") | select(.workflow_run.head_branch == \"$(git branch --show-current)\") | .archive_download_url" \
    | head -n 1)

if [ -z "$ARTIFACT_URL" ]; then
    echo "No artifact is published from the current branch"
    exit 1
fi

echo "Downloading the artifact from $ARTIFACT_URL..."
ARTIFACT_ZIP=artifact.zip
# Use echo | xargs because $ARTIFACT_URL is quoted(like "https://..." as a string), which is invalid form for gh command.
echo "$ARTIFACT_URL" | xargs gh api -H "Accept: application/vnd.github+json" > "$ARTIFACT_ZIP"

unzip -d wheels "$ARTIFACT_ZIP"
