import numpy as np
import pandas as pd
import pytest

from dreamteam import submission


def test_writes_valid_file(tmp_path):
    path = submission.make_submission([1, 2], [0.1, 0.2], out_dir=tmp_path)
    assert pd.read_csv(path).columns.tolist() == ["ID", "target"]


def test_accepts_text_predictions(tmp_path):
    path = submission.make_submission(["a_x", "b_x"], ["Slim", "INVALID"], id_col="test_id",
                                      target_col="des_value", out_dir=tmp_path)
    assert pd.read_csv(path)["des_value"].tolist() == ["Slim", "INVALID"]


@pytest.mark.parametrize("ids,preds", [([1, 1], [0.1, 0.2]), ([1, 2], [0.1, np.nan]), ([1, 2], [0.1, np.inf])])
def test_rejects_bad_submissions(tmp_path, ids, preds):
    with pytest.raises(AssertionError):
        submission.make_submission(ids, preds, out_dir=tmp_path)


def test_follows_sample_order(tmp_path):
    sample = tmp_path / "sample.csv"
    pd.DataFrame({"ID": [3, 1, 2], "target": 0}).to_csv(sample, index=False)
    path = submission.make_submission([1, 2, 3], [10, 20, 30], sample_path=sample, out_dir=tmp_path)
    assert pd.read_csv(path)["ID"].tolist() == [3, 1, 2]
