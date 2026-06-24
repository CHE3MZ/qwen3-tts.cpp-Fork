# Publishing to Scoop and Homebrew

Complete step-by-step guide. Do this when you're ready to publish a release.

---

## Overview

| Package manager | Platform | Install command | Use command |
|----------------|----------|-----------------|-------------|
| Scoop | Windows | `scoop install qwen-tts.cpp` | `qwen-tts` |
| Homebrew | macOS / Linux | `brew install qwen-tts.cpp` | `qwen-tts` |

Package name: **`qwen-tts.cpp`** (mirrors llama.cpp convention)
Binary command: **`qwen-tts`** (shorter, for daily use)

---

## Step 1 — Trigger a Release Build

The GitHub Actions workflow at `.github/workflows/release.yml` fires automatically
when you push a tag matching `release-*`.

```bash
git tag release-1.0
git push origin release-1.0
```

This builds 4 binaries and creates a GitHub Release with these zip assets:
- `qwen-tts-windows-x64-cpu.zip`
- `qwen-tts-windows-x64-cuda.zip`
- `qwen-tts-windows-x64-vulkan.zip`
- `qwen-tts-macos-metal.zip`

Each zip contains `qwen-tts.exe` (Windows) or `qwen-tts` (macOS) + GGML DLLs/dylibs + header.

---

## Step 2 — Get SHA256 Hashes

You need the SHA256 of each zip for the manifests.

**macOS / Linux:**
```bash
# After the release is published, download and hash:
curl -sL https://github.com/CHE3MZ/qwen3-tts.cpp-Fork/releases/download/release-1.0/qwen-tts-macos-metal.zip \
  | shasum -a 256

curl -sL https://github.com/CHE3MZ/qwen3-tts.cpp-Fork/releases/download/release-1.0/qwen-tts-windows-x64-cpu.zip \
  | shasum -a 256
```

**Windows (PowerShell):**
```powershell
$base = "https://github.com/CHE3MZ/qwen3-tts.cpp-Fork/releases/download/release-1.0"
$tmp = "$env:TEMP\hash-check.zip"

Invoke-WebRequest "$base/qwen-tts-windows-x64-cpu.zip" -OutFile $tmp
(Get-FileHash $tmp -Algorithm SHA256).Hash.ToLower()

Invoke-WebRequest "$base/qwen-tts-macos-metal.zip" -OutFile $tmp
(Get-FileHash $tmp -Algorithm SHA256).Hash.ToLower()
```

Alternatively: GitHub shows SHA256 checksums in the release page sidebar once the release is published.

---

## Step 3 — Publish to Scoop

### One-time setup
1. Create a **public** GitHub repo named `scoop-qwen-tts` under your account.
2. Inside that repo, create the folder structure:
   ```
   scoop-qwen-tts/
     bucket/
       qwen-tts.cpp.json
   ```
3. Copy `tools/packages/scoop/qwen-tts.cpp.json` into `bucket/`.

### Per-release update
Edit `bucket/qwen-tts.cpp.json` — update two fields:
```json
{
    "version": "release-1.0",
    "architecture": {
        "64bit": {
            "url": "https://github.com/CHE3MZ/qwen3-tts.cpp-Fork/releases/download/release-1.0/qwen-tts-windows-x64-cpu.zip",
            "hash": "<YOUR_SHA256_HERE>"
        }
    }
}
```

Commit and push to `scoop-qwen-tts`. That's it.

### User experience
```powershell
scoop bucket add qwen-tts.cpp https://github.com/CHE3MZ/scoop-qwen-tts
scoop install qwen-tts.cpp
qwen-tts -m models -t "Hello world" -o hello.wav

# Future updates:
scoop update qwen-tts.cpp
```

---

## Step 4 — Publish to Homebrew

### One-time setup
1. Create a **public** GitHub repo named **`homebrew-qwen-tts`** — the `homebrew-` prefix is mandatory.
2. Inside that repo, create the folder structure:
   ```
   homebrew-qwen-tts/
     Formula/
       qwen-tts.cpp.rb
   ```
3. Copy `tools/packages/homebrew/qwen-tts.cpp.rb` into `Formula/`.

### Per-release update
Edit `Formula/qwen-tts.cpp.rb` — update two fields:
```ruby
version "release-1.0"

on_macos do
  url "https://github.com/CHE3MZ/qwen3-tts.cpp-Fork/releases/download/release-1.0/qwen-tts-macos-metal.zip"
  sha256 "<YOUR_SHA256_HERE>"
end
```

Commit and push to `homebrew-qwen-tts`. Done.

### User experience
```bash
brew tap CHE3MZ/qwen-tts.cpp        # tap name = account/repo-without-homebrew-prefix
brew install qwen-tts.cpp
qwen-tts -m models -t "Hello world" -o hello.wav

# Future updates:
brew upgrade qwen-tts.cpp
```

> **Note on tap naming:** Homebrew strips the `homebrew-` prefix from the repo name.
> So `homebrew-qwen-tts` → tap is `CHE3MZ/qwen-tts` → formula inside is `qwen-tts.cpp.rb`
> → install command is `brew install qwen-tts.cpp`.

---

## Automating future releases (optional)

Once you have a release cadence, you can automate the hash updates:

**Scoop:** Add a GitHub Actions workflow to `scoop-qwen-tts` using `scoop-autoupdate`
that runs on a schedule and checks for new releases in `qwen3-tts.cpp-Fork`.

**Homebrew:** Use `brew bump-formula-pr` locally after each release:
```bash
brew bump-formula-pr --version release-1.1 \
  --url https://github.com/CHE3MZ/qwen3-tts.cpp-Fork/releases/download/release-1.1/qwen-tts-macos-metal.zip \
  --sha256 <NEW_HASH> \
  CHE3MZ/qwen-tts.cpp/qwen-tts.cpp
```

---

## File reference

| File | Purpose |
|------|---------|
| `tools/packages/scoop/qwen-tts.cpp.json` | Scoop manifest template (copy to bucket repo) |
| `tools/packages/homebrew/qwen-tts.cpp.rb` | Homebrew formula template (copy to tap repo) |
| `.github/workflows/release.yml` | Automated release builder (tag `release-X.Y` to trigger) |
