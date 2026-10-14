"""The inference adapter aligns user input with the training schema."""

from app.utils.inference import build_inference_dataframe

EXPECTED = ["A", "B", "C"]


def test_columns_follow_training_order():
    df = build_inference_dataframe({"C": 3, "A": 1}, EXPECTED)
    assert list(df.columns) == EXPECTED
    assert len(df) == 1


def test_missing_columns_are_empty():
    df = build_inference_dataframe({"A": 1}, EXPECTED)
    assert df.loc[0, "A"] == 1
    assert df[["B", "C"]].isna().all().all()


def test_unknown_inputs_are_dropped():
    df = build_inference_dataframe({"A": 1, "NOT_A_FEATURE": 9}, EXPECTED)
    assert "NOT_A_FEATURE" not in df.columns
