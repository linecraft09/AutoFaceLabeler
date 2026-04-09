import aflutils.video_utils as VU
from core.storage.video_store import VideoStore

if __name__ == "__main__":
    video_store = VideoStore(db_path="D:/WorkDir/AutoFaceLabeler/data/videos.db")

    stale_video_paths = video_store.get_file_paths_by_excluded_statuses(
        excluded_statuses=("downloaded", "v2_passed")
    )
    if stale_video_paths:
        print(f"Cleaning stale videos: {len(stale_video_paths)} files")
        VU.remove_videos(stale_video_paths)
