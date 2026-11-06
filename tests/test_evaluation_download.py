import numpy as np
import pytest

from modelgate.download import unpack_verified_archive
from modelgate.evaluation import block_bootstrap_intervals


def test_source_download_rejects_wrong_bytes_before_unpacking(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        unpack_verified_archive(b"untrusted source", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_block_bootstrap_reports_known_perfect_ranking_and_brier():
    y = np.tile([0, 1], 100)
    p = np.where(y == 1, 0.8, 0.2)
    report = block_bootstrap_intervals(y, p, repetitions=30, block_size=20)
    assert report["average_precision"] == [1.0, 1.0]
    assert report["brier"] == pytest.approx([0.04, 0.04])
    assert report["method"] == "moving_block_percentile"
