from persian_intent.baseline import build_pipeline, classification_metrics


def test_classification_metrics_include_macro_and_weighted_f1():
    result = classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        intent_names=["alarm_set", "play_music"],
    )

    assert result["n_examples"] == 4
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert 0.0 <= result["weighted_f1"] <= 1.0
    assert "alarm_set" in result["classification_report"]
    assert "play_music" in result["classification_report"]


def test_pipeline_can_fit_small_persian_text_sample():
    texts = [
        "موسیقی پخش کن",
        "یک آهنگ پخش کن",
        "آهنگ بعدی را پخش کن",
        "صدای موسیقی را کم کن",
        "برای فردا زنگ بگذار",
        "ساعت هفت مرا بیدار کن",
        "زنگ ساعت را تنظیم کن",
        "فردا صبح زنگ بزن",
    ]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    model = build_pipeline(max_features=1_000, max_iter=100)
    model.fit(texts, labels)

    predictions = model.predict(["موسیقی را پخش کن", "برای صبح زنگ بگذار"])
    assert len(predictions) == 2
    assert set(predictions).issubset({0, 1})
