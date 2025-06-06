# from typing import Optional

# import numpy as np
# import pytest
# import torch


# @pytest.fixture(scope="module")
# def tracker() -> BeatTracker:
#     return BeatTracker()


# @pytest.fixture(scope="module")
# def audio_path() -> str:
#     return "tests/audio_example.mp3"


# SAMPLE_RATE = 44110


# def test_load_audio(tracker: BeatTracker, audio_path: str) -> None:
#     audio_signal, _ = tracker.load_audio("tests/audio_example.mp3")

#     assert isinstance(audio_signal, torch.Tensor)
#     assert len(audio_signal) > 0
#     assert audio_signal.ndim == 1


# def test_beat_tracking(tracker: BeatTracker, audio_path: str) -> None:
#     result = tracker.beat_track(audio_path)

#     assert isinstance(result.beats, np.ndarray)
#     assert isinstance(result.downbeats, np.ndarray)
#     assert len(result.beats) > 10
#     assert len(result.downbeats) > 5

#     # Check that the metrics were computed
#     assert result.metrics is not None
#     assert isinstance(result.metrics.tempo, float)
#     assert isinstance(result.metrics.strength, float)
#     assert isinstance(result.metrics.stability, float)


# def test_beat_tracking_on_signal(tracker: BeatTracker, audio_path: str) -> None:
#     audio_signal, sr = tracker.load_audio(audio_path)
#     result = tracker.beat_track_from_audio(audio_signal, sr)

#     assert isinstance(result.beats, np.ndarray)
#     assert isinstance(result.downbeats, np.ndarray)
#     assert len(result.beats) > 10
#     assert len(result.downbeats) > 5

#     # Check that the metrics were computed
#     assert result.metrics is not None
#     assert isinstance(result.metrics.tempo, float)
#     assert isinstance(result.metrics.strength, float)
#     assert isinstance(result.metrics.stability, float)


# @pytest.mark.parametrize(
#     "audio_signal, len_expected_beats, len_expected_downbeats, expected_metrics",
#     [
#         (torch.zeros(SAMPLE_RATE), 0, 0, None),  # No beat
#         (torch.concat((torch.zeros(SAMPLE_RATE), torch.ones(1), torch.zeros(SAMPLE_RATE))), 1, 0, None),  # One beat
#     ],
# )
# def test_beat_tracking_various_cases(
#     tracker: BeatTracker,
#     audio_signal: torch.Tensor,
#     len_expected_beats: int,
#     len_expected_downbeats: int,
#     expected_metrics: Optional[BeatMetrics],
# ) -> None:
#     result = tracker.beat_track_from_audio(audio_signal, SAMPLE_RATE)

#     assert isinstance(result.beats, np.ndarray)
#     assert isinstance(result.downbeats, np.ndarray)
#     assert len(result.beats) == len_expected_beats
#     assert len(result.downbeats) == len_expected_downbeats
#     assert result.metrics == expected_metrics
