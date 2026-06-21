"""
_hf_download.py -- called by download-model.bat and download-tokenizer.bat
Usage: python _hf_download.py <repo_id> <local_dir> [hf_token]
Retries up to 5 times with exponential backoff on network errors.
"""
import sys, time

def main():
    if len(sys.argv) < 3:
        print("Usage: _hf_download.py <repo_id> <local_dir> [hf_token]")
        sys.exit(1)

    repo_id   = sys.argv[1]
    local_dir = sys.argv[2]
    token     = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None

    from huggingface_hub import snapshot_download

    delay = 5.0
    for attempt in range(1, 6):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                token=token,
                resume_download=True,
            )
            print(f"[ok] Download complete: {repo_id}")
            return
        except Exception as e:
            msg = str(e)
            is_network = any(k in msg for k in (
                "ConnectionReset", "ConnectionError", "ProtocolError",
                "10054", "RemoteDisconnected", "LocalEntryNotFound",
            ))
            if attempt == 5 or not is_network:
                print(f"[error] Download failed after {attempt} attempt(s): {e}")
                sys.exit(1)
            print(f"  [warn] attempt {attempt}/5 failed: {type(e).__name__}. Retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)

if __name__ == "__main__":
    main()
