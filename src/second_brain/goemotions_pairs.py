# Valence/arousal coordinates for all 27 GoEmotions categories + neutral.
# Source: NRC-VAD Lexicon (Mohammad, 2018, ACL -- "Obtaining Reliable Human
# Ratings of Valence, Arousal, and Dominance for 20,000 English Words"),
# downloaded and looked up directly against the category label words --
# every one of the 28 labels has a direct entry, no approximation needed.
# Dominance is available in the source lexicon but unused here; this project
# follows Russell's circumplex (valence/arousal only).
VA_LOOKUP: dict[str, tuple[float, float]] = {
    "admiration": (0.969, 0.583),
    "amusement": (0.929, 0.837),
    "anger": (0.167, 0.865),
    "annoyance": (0.167, 0.718),
    "approval": (0.854, 0.460),
    "caring": (0.635, 0.469),
    "confusion": (0.255, 0.667),
    "curiosity": (0.750, 0.755),
    "desire": (0.896, 0.692),
    "disappointment": (0.115, 0.490),
    "disapproval": (0.085, 0.551),
    "disgust": (0.052, 0.775),
    "embarrassment": (0.143, 0.685),
    "excitement": (0.896, 0.684),
    "fear": (0.073, 0.840),
    "gratitude": (0.885, 0.441),
    "grief": (0.070, 0.640),
    "joy": (0.980, 0.824),
    "love": (1.000, 0.519),
    "nervousness": (0.163, 0.915),
    "optimism": (0.949, 0.565),
    "pride": (0.729, 0.634),
    "realization": (0.554, 0.510),
    "relief": (0.844, 0.278),
    "remorse": (0.103, 0.673),
    "sadness": (0.052, 0.288),
    "surprise": (0.875, 0.875),
    "neutral": (0.469, 0.184),
}

GOEMOTIONS_CATEGORIES = list(VA_LOOKUP.keys())

MAX_VA_DISTANCE = 2 ** 0.5  # valence and arousal each range [0,1] -> max Euclidean distance is sqrt(2)


def euclidean_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5


def base_similarity_from_distance(distance: float) -> float:
    """Linear conversion from VA distance to a [0,1] similarity, matching
    CosineSimilarityLoss's expected target range. 0 distance -> 1.0
    similarity, max distance (sqrt(2)) -> 0.0 similarity."""
    return 1 - (distance / MAX_VA_DISTANCE)


def shares_category(labels_a: list[str], labels_b: list[str]) -> bool:
    """Two examples are the 'same category' if their label sets overlap at
    all (non-empty intersection) -- extends the same/different-label idea
    to GoEmotions' multi-label examples (~17% of rows) without arbitrarily
    picking one label per example."""
    return not set(labels_a).isdisjoint(labels_b)


DEFAULT_DIFFERENT_LABEL_SIMILARITY_CAP = 0.7


def compute_target_similarity(
    labels_a: list[str], labels_b: list[str], cap: float = DEFAULT_DIFFERENT_LABEL_SIMILARITY_CAP
) -> float:
    """Target similarity for a contrastive training pair: pure VA-distance
    similarity for same-category pairs, capped for different-category pairs
    so VA-close-but-distinct emotions (e.g. anger/fear) are pulled somewhat
    close but never fully merged with true same-category pairs. The cap
    only ever pulls a similarity DOWN -- a pair already below the cap
    (e.g. VA-distant emotions like joy/sadness) is left untouched."""
    distance = euclidean_distance(get_va_coordinates(labels_a), get_va_coordinates(labels_b))
    base_similarity = base_similarity_from_distance(distance)

    if shares_category(labels_a, labels_b):
        return base_similarity
    return min(base_similarity, cap)


def get_va_coordinates(labels: list[str]) -> tuple[float, float]:
    """VA point for an example: the lookup value for a single label, or the
    simple average across labels for a multi-label example (~17% of
    GoEmotions rows carry 2+ labels)."""
    if not labels:
        raise ValueError("labels must be non-empty")

    valences = [VA_LOOKUP[label][0] for label in labels]
    arousals = [VA_LOOKUP[label][1] for label in labels]
    return (sum(valences) / len(labels), sum(arousals) / len(labels))
