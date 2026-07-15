PERSONAL_HYPOTHESIS_LABEL = "a personal or emotional topic"
TASK_HYPOTHESIS_LABEL = "a task, question, or request for help"


def build_classifier(model_name: str = "facebook/bart-large-mnli"):
    import torch
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline("zero-shot-classification", model=model_name, device=device)


def classify_text(text: str, classifier, title: str | None = None) -> dict:
    """Classify a message as personal or task. Pass title=None for the primary
    (text-only) metric; pass the conversation title for the title+text ablation."""
    input_text = text if title is None else f"{title}. {text}"

    result = classifier(
        input_text,
        candidate_labels=[PERSONAL_HYPOTHESIS_LABEL, TASK_HYPOTHESIS_LABEL],
        hypothesis_template="This message is about {}.",
    )

    top_label = result["labels"][0]
    top_score = result["scores"][0]
    normalized = "personal" if top_label == PERSONAL_HYPOTHESIS_LABEL else "task"
    return {"label": normalized, "score": top_score, "raw_label": top_label}
