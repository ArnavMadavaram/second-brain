def compute_classification_metrics(
    true_labels: list[str], predicted_labels: list[str], positive_label: str = "personal"
) -> dict:
    tp = fp = fn = tn = 0
    confusion: dict[str, dict[str, int]] = {}

    for true, pred in zip(true_labels, predicted_labels):
        confusion.setdefault(true, {}).setdefault(pred, 0)
        confusion[true][pred] += 1

        if true == positive_label and pred == positive_label:
            tp += 1
        elif true != positive_label and pred == positive_label:
            fp += 1
        elif true == positive_label and pred != positive_label:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(true_labels) if true_labels else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "n": len(true_labels),
    }


def build_eval_pairs(
    labels: list[dict], predictions: dict[str, str], split: str = "test"
) -> tuple[list[str], list[str]]:
    filtered = [r for r in labels if r["split"] == split and r["label"] != "exclude"]

    true_labels = []
    predicted_labels = []
    for r in filtered:
        true_labels.append(r["label"])
        predicted_labels.append(predictions[r["message_id"]])  # KeyError if missing -- intentional

    return true_labels, predicted_labels
