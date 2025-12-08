# Release Process

This document explains how to create releases for NeuroShard.

## Overview

NeuroShard uses GitHub tags to trigger automated builds and releases. When you create a tag matching the pattern `v*` (e.g., `v1.0.0`), GitHub Actions will automatically:

1. Build binaries for Linux, Windows, and macOS
2. Create a GitHub Release with all three binaries attached
3. Make the release available on the download page

## Creating a Release

### Step 1: Create and Push a Tag

Tags should follow semantic versioning (e.g., `v1.0.0`, `v1.2.3`, `v2.0.0-beta`).

**Using Git commands:**

```bash
# Create an annotated tag (recommended)
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push the tag to GitHub
git push origin v1.0.0

# Or push all tags at once
git push --tags
```

**Using GitHub UI:**

1. Go to your repository on GitHub
2. Click on "Releases" → "Create a new release"
3. Click "Choose a tag" and type your tag name (e.g., `v1.0.0`)
4. Click "Create new tag: v1.0.0 on publish"
5. Fill in the release title and description
6. Click "Publish release"

**Note:** If you create the tag through the GitHub UI, the workflow will automatically trigger when you publish the release.

### Step 2: Monitor the Build

1. Go to the "Actions" tab in your GitHub repository
2. You'll see a workflow run titled "Build NeuroShard Nodes" triggered by your tag
3. Wait for all three build jobs (Linux, Windows, macOS) to complete
4. The release will be created automatically once all builds succeed

### Step 3: Verify the Release

1. Go to "Releases" in your GitHub repository
2. You should see your new release with all three binaries attached:
   - `NeuroShardNode` (Linux)
   - `NeuroShardNode.exe` (Windows)
   - `NeuroShardNode_Mac.zip` (macOS)

## Tag Naming Convention

- Use semantic versioning: `vMAJOR.MINOR.PATCH`
- Examples: `v1.0.0`, `v1.2.3`, `v2.0.0`
- Pre-release versions: `v1.0.0-beta`, `v1.0.0-rc1`
- Tags must start with `v` to trigger the workflow

## Website Integration

The website's download page automatically fetches the latest release from GitHub. Once a release is published:

1. The `/api/downloads/latest` endpoint will fetch the latest release from GitHub
2. The download page will display the latest version and download links
3. Users can download the appropriate binary for their platform

If no releases are found on GitHub, the website will fall back to mock data for development purposes.

## Deleting and Recreating a Tag

If you need to delete a tag (e.g., if the release failed or you want to fix something):

**Delete the tag locally:**
```bash
git tag -d v0.0.1
```

**Delete the tag on GitHub (remote):**
```bash
git push origin --delete v0.0.1
# Or alternatively:
git push origin :refs/tags/v0.0.1
```

**If a GitHub Release was created, you'll also need to delete it:**
1. Go to your repository on GitHub
2. Click on "Releases"
3. Find the release associated with the tag
4. Click "Delete release" (you may need to delete the release before deleting the tag)

**After deleting, recreate the tag:**
```bash
# Create the tag again
git tag -a v0.0.1 -m "Release version 0.0.1"

# Push it to GitHub
git push origin v0.0.1
```

This will trigger the GitHub Actions workflow again.

## Troubleshooting

### Build Fails

- Check the "Actions" tab for error messages
- Verify that `neuroshard.spec` is correctly configured

### Release Not Created

- Ensure the tag matches the pattern `v*`
- Check that all three build jobs completed successfully
- Verify that `GITHUB_TOKEN` has the necessary permissions (this is automatic for public repos)

### Website Not Showing Latest Release

- Verify the release exists on GitHub
- Check that the repository name in `website/api/downloads.py` matches your actual GitHub repository
- Check the backend logs for any API errors

## Manual Release (Alternative)

If you need to create a release manually without triggering the build:

1. Build the binaries locally using the build scripts
2. Go to GitHub Releases
3. Create a new release and manually upload the binaries

However, using tags is the recommended approach as it ensures consistent builds across all platforms.

