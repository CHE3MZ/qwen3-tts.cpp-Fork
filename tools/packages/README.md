# Package Distribution

Package name: **`qwen-tts.cpp`**
Binary command: **`qwen-tts`**

This mirrors the llama.cpp convention: install with the full name (`qwen-tts.cpp`),
run with the short command (`qwen-tts`).

---

## Scoop (Windows)

```powershell
# Add the bucket (hosted at https://github.com/CHE3MZ/scoop-qwen-tts)
scoop bucket add qwen-tts.cpp https://github.com/CHE3MZ/scoop-qwen-tts

# Install
scoop install qwen-tts.cpp

# Use
qwen-tts -m models -t "Hello world" -o hello.wav
```

The bucket repo (`CHE3MZ/scoop-qwen-tts`) must contain `bucket/qwen-tts.cpp.json`,
copied from `package/scoop/qwen-tts.cpp.json` in this repo.

After each release, update the `version` and `hash` fields in the manifest.
Scoop's `checkver` + `autoupdate` can automate this with a GitHub Actions job
in the bucket repo.

---

## Homebrew (macOS / Linux)

```bash
# Add the tap (hosted at https://github.com/CHE3MZ/homebrew-qwen-tts)
brew tap CHE3MZ/qwen-tts.cpp

# Install
brew install qwen-tts.cpp

# Use
qwen-tts -m models -t "Hello world" -o hello.wav
```

The tap repo (`CHE3MZ/homebrew-qwen-tts`) must contain `Formula/qwen-tts.cpp.rb`,
copied from `package/homebrew/qwen-tts.cpp.rb` in this repo.

After each release, update the `version` and `sha256` fields in the formula.
Homebrew's `brew bump-formula-pr` can automate this.

---

## Release process

1. Tag the release:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. The `.github/workflows/release.yml` workflow triggers automatically,
   builds all four backend zips, and creates a GitHub Release.

3. Download the release zip SHA256 hashes:
   ```bash
   # macOS
   curl -sL https://github.com/CHE3MZ/qwen3-tts.cpp-Fork/releases/download/v1.0.0/qwen-tts-macos-metal.zip | sha256sum
   # Windows
   certutil -hashfile qwen-tts-windows-x64-cpu.zip SHA256
   ```

4. Update `package/scoop/qwen-tts.cpp.json` — replace `PLACEHOLDER_SHA256` and `version`.
5. Update `package/homebrew/qwen-tts.cpp.rb` — replace `PLACEHOLDER_SHA256_MACOS` and `version`.
6. Push the updated manifests to the respective bucket/tap repos.

---

## File layout

```
package/
  scoop/
    qwen-tts.cpp.json      ← Scoop manifest (copy to bucket repo)
  homebrew/
    qwen-tts.cpp.rb        ← Homebrew formula (copy to tap repo)
  README.md                ← This file
```
