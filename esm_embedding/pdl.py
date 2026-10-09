"""Parallel range-request downloader for throttled-per-connection links.

Usage: python pdl.py URL OUT [NCONN]
"""
import sys, os, threading, time
import urllib.request

URL, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 32
CHUNK = 8 * 1024 * 1024  # 8 MiB work units


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url, start, end, retries=8):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"range {start}-{end} failed after {retries} tries")


def main():
    total = head_size(URL)
    print(f"total {total/1e6:.1f} MB, {N} connections, chunk {CHUNK//1024} KiB")
    part = OUT + ".part"
    # preallocate
    if not os.path.exists(part) or os.path.getsize(part) != total:
        with open(part, "wb") as f:
            f.truncate(total)
    state = OUT + ".state"
    done = set()
    if os.path.exists(state):
        done = set(int(x) for x in open(state).read().split())
    units = [(i, i * CHUNK, min((i + 1) * CHUNK, total) - 1)
             for i in range((total + CHUNK - 1) // CHUNK)]
    todo = [u for u in units if u[0] not in done]
    print(f"{len(done)} units done, {len(todo)} to go")
    lock = threading.Lock()
    sf = open(state, "a")
    progress = [0, time.time()]

    def worker(queue):
        while True:
            with lock:
                if not queue:
                    return
                idx, start, end = queue.pop(0)
            try:
                data = fetch_range(URL, start, end)
                with lock:
                    with open(part, "r+b") as f:
                        f.seek(start)
                        f.write(data)
                    sf.write(f"{idx}\n")
                    sf.flush()
                    progress[0] += len(data)
                    el = time.time() - progress[1]
                    if progress[0] > 20e6 or not queue:
                        print(f"  {progress[0]/1e6:.0f} MB "
                              f"({progress[0]/el/1e6:.2f} MB/s)", flush=True)
                        progress[0], progress[1] = 0, time.time()
            except Exception as e:
                print(f"  unit {idx} failed: {e}", flush=True)
                with lock:
                    queue.append((idx, start, end))

    q = list(todo)
    threads = [threading.Thread(target=worker, args=(q,), daemon=True)
               for _ in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not q:
        os.rename(part, OUT)
        os.remove(state)
        print("DONE", OUT)
    else:
        print("INCOMPLETE — rerun to resume")


if __name__ == "__main__":
    main()
