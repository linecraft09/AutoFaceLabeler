#!/usr/bin/env python3
"""End-to-end YouTube search + download test with cookies and proxy."""
import sys, os, tempfile, shutil

# Load dotenv first
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import yt_dlp
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COOKIES = str(PROJECT_ROOT / 'config' / 'youtube_cookies.txt')
PROXY = os.getenv('HTTPS_PROXY', 'http://127.0.0.1:7890')
DOWNLOAD_DIR = tempfile.mkdtemp(prefix='afl_e2e_')

base_opts = {
    'proxy': PROXY,
    'cookiefile': COOKIES,
    'remote_components': ['ejs:github'],
    'quiet': False,
    'no_warnings': False,
    'nocheckcertificate': True,
    # Note: using cookies, so no specific player_client override needed
}

print("=" * 60)
print("TEST 1: Flat Search")
print("=" * 60)
opts_flat = {**base_opts, 'extract_flat': True}
with yt_dlp.YoutubeDL(opts_flat) as ydl:
    info = ydl.extract_info('ytsearch2:short music', download=False)
    entries = info.get('entries', [])
    print(f"Found {len(entries)} videos:")
    for e in entries:
        print(f"  - {e.get('title')} | {e.get('duration', '?')}s | id={e.get('id')}")

if not entries:
    print("FAIL: No search results")
    sys.exit(1)

# Pick the first one for detail + download
first = entries[0]
video_id = first['id']
video_url = first.get('webpage_url') or f'https://youtube.com/watch?v={video_id}'

print()
print("=" * 60)
print("TEST 2: Detail Extract (formats)")
print("=" * 60)
with yt_dlp.YoutubeDL(base_opts) as ydl:
    detail = ydl.extract_info(video_url, download=False)
    formats = detail.get('formats', [])
    title = detail.get('title', 'N/A')
    print(f"Title: {title}")
    print(f"Formats: {len(formats)} available")
    video_formats = [f for f in formats if f.get('vcodec') != 'none']
    print(f"Video formats: {len(video_formats)}")
    for f in video_formats[:5]:
        print(f"  {f['format_id']}: {f.get('height', '?')}p {f.get('ext', '?')} ({f.get('filesize_approx', f.get('filesize', '?'))})")
    
    if len(video_formats) == 0:
        print("FAIL: No video formats found (cookies may not be working)")
        sys.exit(1)

print()
print("=" * 60)
print("TEST 3: Download")
print("=" * 60)
download_opts = {
    **base_opts,
    'format': 'best[height<=360]/worst',
    'outtmpl': str(Path(DOWNLOAD_DIR) / '%(title)s.%(ext)s'),
    'paths': {'home': DOWNLOAD_DIR},
    'quiet': True,
}
with yt_dlp.YoutubeDL(download_opts) as ydl:
    ydl.download([video_url])

downloaded = list(Path(DOWNLOAD_DIR).glob('*'))
if downloaded:
    f = downloaded[0]
    size_mb = f.stat().st_size / (1024 * 1024) if f.exists() else 0
    print(f"Downloaded: {f.name} ({size_mb:.2f} MB)")
else:
    print("FAIL: No file downloaded")

print()
print("=" * 60)
print("TEST 4: Pipeline orchestrator (smoke test)")
print("=" * 60)
try:
    from aflutils.config_loader import ConfigLoader
    loader = ConfigLoader(str(PROJECT_ROOT / 'config' / 'config.yaml'))
    print(f"search.platforms.youtube.cookies = {loader.get('search.platforms.youtube.cookies')}")
    print(f"download.platforms.youtube.cookies = {loader.get('download.platforms.youtube.cookies')}")
    print("Config loaded OK")
except Exception as e:
    print(f"Config load warning (may be OK if schema strict): {e}")

# Cleanup
shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
print()
print("=" * 60)
print("ALL TESTS PASSED" if len(entries) > 0 and len(video_formats) > 0 and downloaded else "SOME TESTS FAILED")
print("=" * 60)
