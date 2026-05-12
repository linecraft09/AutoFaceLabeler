import os
import subprocess

import pytest

import aflutils.video_utils as VU


def _completed(cmd, returncode=0, stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)


def test_sample_video_per_sec_falls_back_to_libx264_and_forces_8bit(monkeypatch, tmp_path):
    input_path = tmp_path / "input.mkv"
    input_path.write_bytes(b"video")
    calls = []

    monkeypatch.setattr(VU, "_can_try_h264_nvenc", lambda: True)
    monkeypatch.setattr(VU, "_run_command", lambda cmd, timeout=300, use_cuda_libs=False: _completed(cmd))

    def fake_run_ffmpeg(cmd, timeout=300, use_cuda_libs=False):
        calls.append((cmd, use_cuda_libs))
        if "h264_nvenc" in cmd:
            output_path = cmd[-1]
            open(output_path, "wb").close()
            return _completed(cmd, returncode=218, stderr="10 bit encode not supported")
        output_path = cmd[-1]
        with open(output_path, "wb") as f:
            f.write(b"sample")
        return _completed(cmd)

    monkeypatch.setattr(VU, "_run_ffmpeg", fake_run_ffmpeg)

    output_path = VU.sample_video_per_sec(str(input_path))

    assert os.path.exists(output_path)
    assert len(calls) == 2
    assert "h264_nvenc" in calls[0][0]
    assert calls[0][1] is True
    assert "libx264" in calls[1][0]
    assert calls[1][1] is False
    assert "fps=1,scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" in calls[0][0]
    assert "fps=1,scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" in calls[1][0]
    assert "-hwaccel" not in calls[0][0]


def test_sample_video_per_sec_uses_software_when_nvenc_unavailable(monkeypatch, tmp_path):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    calls = []

    monkeypatch.setattr(VU, "_can_try_h264_nvenc", lambda: False)
    monkeypatch.setattr(VU, "_run_command", lambda cmd, timeout=300, use_cuda_libs=False: _completed(cmd))

    def fake_run_ffmpeg(cmd, timeout=300, use_cuda_libs=False):
        calls.append((cmd, use_cuda_libs))
        with open(cmd[-1], "wb") as f:
            f.write(b"sample")
        return _completed(cmd)

    monkeypatch.setattr(VU, "_run_ffmpeg", fake_run_ffmpeg)

    output_path = VU.sample_video_per_sec(str(input_path))

    assert os.path.exists(output_path)
    assert len(calls) == 1
    assert "libx264" in calls[0][0]
    assert "h264_nvenc" not in calls[0][0]


def test_clip_video_uses_copy_then_software_fallback_with_mkv_output(monkeypatch, tmp_path):
    input_path = tmp_path / "input.webm"
    input_path.write_bytes(b"video")
    copy_calls = []
    software_calls = []

    class FakeCapture:
        def get(self, prop):
            return 10.0

        def release(self):
            pass

    monkeypatch.setattr(VU.cv2, "VideoCapture", lambda path: FakeCapture())

    def fake_clip_with_copy(video_path, start, end, output):
        copy_calls.append((video_path, start, end, output))
        return False

    def fake_clip_with_software(video_path, start, end, output):
        software_calls.append((video_path, start, end, output))
        return True

    monkeypatch.setattr(VU, "_clip_with_copy", fake_clip_with_copy)
    monkeypatch.setattr(VU, "_clip_with_software", fake_clip_with_software)

    outputs = VU.clip_video(str(input_path), [(0, 1)], unit="second", max_workers=1)

    assert outputs == [str(tmp_path / "input_clipped_0.mkv")]
    assert copy_calls
    assert software_calls


def test_probe_video_parses_video_and_audio_metadata(monkeypatch):
    payload = """{
      "streams": [
        {
          "codec_type": "video",
          "codec_name": "vp9",
          "pix_fmt": "yuv420p10le",
          "width": 1920,
          "height": 1078,
          "avg_frame_rate": "60/1",
          "duration": "99.01",
          "nb_frames": "5941",
          "color_space": "bt2020nc",
          "color_transfer": "arib-std-b67",
          "color_primaries": "bt2020"
        },
        {"codec_type": "audio", "codec_name": "opus"}
      ],
      "format": {"duration": "99.01"}
    }"""

    def fake_run_command(cmd, timeout=300, use_cuda_libs=False):
        return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr="")

    monkeypatch.setattr(VU, "_run_command", fake_run_command)

    info = VU.probe_video("video.mkv")

    assert info.codec_name == "vp9"
    assert info.pix_fmt == "yuv420p10le"
    assert info.fps == pytest.approx(60.0)
    assert info.duration == pytest.approx(99.01)
    assert info.frame_count == 5941
    assert info.has_audio is True
