from src.preprocess import FEATURES, TARGET, load_data


def test_load_data_keeps_only_safe_columns():
    df = load_data("data/telco.csv")
    assert set(df.columns) == set(FEATURES + [TARGET])


def test_no_leakage_columns():
    df = load_data("data/telco.csv")
    for col in ["ChurnScore", "CustomerStatus", "ChurnReason", "CLTV", "SatisfactionScore"]:
        assert col not in df.columns


def test_target_is_binary():
    df = load_data("data/telco.csv")
    assert set(df[TARGET].unique()) == {0, 1}