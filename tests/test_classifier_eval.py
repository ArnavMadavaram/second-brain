import pytest

from second_brain.classifier_eval import build_eval_pairs, compute_classification_metrics


def test_perfect_predictions_give_perfect_metrics():
    true_labels = ["personal", "personal", "task", "task"]
    predicted = ["personal", "personal", "task", "task"]

    metrics = compute_classification_metrics(true_labels, predicted)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["n"] == 4


def test_metrics_with_false_positive_and_false_negative():
    # true:      personal, personal, task,     task
    # predicted: personal, task,     personal, task
    # tp=1 (idx0), fn=1 (idx1, missed a personal), fp=1 (idx2, wrongly called personal), tn=1 (idx3)
    true_labels = ["personal", "personal", "task", "task"]
    predicted = ["personal", "task", "personal", "task"]

    metrics = compute_classification_metrics(true_labels, predicted)

    assert metrics["precision"] == pytest.approx(0.5)  # tp / (tp + fp) = 1/2
    assert metrics["recall"] == pytest.approx(0.5)      # tp / (tp + fn) = 1/2
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)     # 2 correct out of 4


def test_confusion_matrix_structure():
    true_labels = ["personal", "task"]
    predicted = ["task", "task"]

    metrics = compute_classification_metrics(true_labels, predicted)

    assert metrics["confusion_matrix"]["personal"]["task"] == 1
    assert metrics["confusion_matrix"]["task"]["task"] == 1


def test_zero_predicted_positives_gives_zero_precision_not_divide_by_zero():
    true_labels = ["personal", "task"]
    predicted = ["task", "task"]

    metrics = compute_classification_metrics(true_labels, predicted)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_build_eval_pairs_filters_to_test_split_and_excludes_ambiguous():
    labels = [
        {"message_id": "m1", "split": "test", "label": "personal"},
        {"message_id": "m2", "split": "dev", "label": "task"},       # wrong split, excluded
        {"message_id": "m3", "split": "test", "label": "exclude"},   # ambiguous, excluded
        {"message_id": "m4", "split": "test", "label": "task"},
    ]
    predictions = {"m1": "personal", "m2": "personal", "m3": "task", "m4": "personal"}

    true_labels, predicted_labels = build_eval_pairs(labels, predictions, split="test")

    assert true_labels == ["personal", "task"]
    assert predicted_labels == ["personal", "personal"]


def test_build_eval_pairs_raises_on_missing_prediction():
    labels = [{"message_id": "m1", "split": "test", "label": "personal"}]
    predictions = {}

    with pytest.raises(KeyError):
        build_eval_pairs(labels, predictions, split="test")
