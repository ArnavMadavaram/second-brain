# Attempt 2 (current default): reworded after reading Attempt 1's actual errors.
# Attempt 1 used "a personal or emotional topic" / "a task, question, or request
# for help" -- diagnosis showed false negatives cluster on question-form personal
# messages ("Can I legally...") being pulled toward "task" just because they're
# phrased as requests (true of nearly every message), and false positives cluster
# on first-person narrative tone ("my car", "I upgraded") regardless of content.
# These labels anchor on content domain instead of speech-act form.
PERSONAL_HYPOTHESIS_LABEL = "an expression of the person's own feelings, relationships, health, or personal life circumstances"
TASK_HYPOTHESIS_LABEL = "a factual, technical, or academic question unrelated to the person's personal life or feelings"

HYPOTHESIS_TEMPLATE = "This message is {}."


def build_classifier(model_name: str = "facebook/bart-large-mnli"):
    import torch
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline("zero-shot-classification", model=model_name, device=device)


def classify_text(
    text: str,
    classifier,
    title: str | None = None,
    personal_label: str = PERSONAL_HYPOTHESIS_LABEL,
    task_label: str = TASK_HYPOTHESIS_LABEL,
    hypothesis_template: str = HYPOTHESIS_TEMPLATE,
) -> dict:
    """Classify a message as personal or task. Pass title=None for the primary
    (text-only) metric; pass the conversation title for the title+text ablation.
    personal_label/task_label/hypothesis_template are overridable so different
    hypothesis-phrasing attempts can be run and compared explicitly."""
    input_text = text if title is None else f"{title}. {text}"

    result = classifier(
        input_text,
        candidate_labels=[personal_label, task_label],
        hypothesis_template=hypothesis_template,
    )

    top_label = result["labels"][0]
    top_score = result["scores"][0]
    normalized = "personal" if top_label == personal_label else "task"
    return {"label": normalized, "score": top_score, "raw_label": top_label}
