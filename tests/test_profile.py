import pandas as pd

from aml.profile_data import profile


def test_profile_reports_shape_missing_and_imbalance():
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2023-01-01", periods=4, freq="h"),
            "Sender_account": [1, 2, 3, 4],
            "Receiver_account": [2, 3, 4, 1],
            "Amount": [10.0, 20.0, 30.0, 40.0],
            "Payment_type": pd.Categorical(["ACH", "ACH", "Cheque", "ACH"]),
            "Is_laundering": [0, 0, 0, 1],
        }
    )
    p = profile(df, "Is_laundering")
    assert p["shape"] == [4, 6] and p["duplicate_rows"] == 0
    assert p["target"]["positives"] == 1 and p["target"]["imbalance_ratio_neg_to_pos"] == 3.0
    assert p["categorical_cardinality"] == {"Payment_type": 2}
    assert p["missing"]["Amount"] == 0
