import random

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


def group_by_primary_category(examples: list[dict]) -> dict[str, list[dict]]:
    """Group examples by their first ('primary') label. Used only for
    anchor-selection bookkeeping (capping, category buckets) -- the exact
    training target for any generated pair always uses the full multi-label
    set via compute_target_similarity, regardless of which bucket an example
    was drawn from."""
    grouped: dict[str, list[dict]] = {}
    for example in examples:
        primary = example["labels"][0]
        grouped.setdefault(primary, []).append(example)
    return grouped


def cap_per_category(grouped: dict[str, list[dict]], cap: int, seed: int = 400) -> list[dict]:
    """Flatten grouped examples into one anchor list. Categories at or below
    `cap` keep every example; categories above `cap` are downsampled to
    exactly `cap` (random subset) so common categories (e.g. neutral at
    32.8% of the corpus) don't dominate the anchor pool and starve rarer,
    often more emotionally distinctive categories (e.g. grief at 0.2%) of
    training signal."""
    rng = random.Random(seed)
    result: list[dict] = []
    for category_examples in grouped.values():
        if len(category_examples) <= cap:
            result.extend(category_examples)
        else:
            result.extend(rng.sample(category_examples, cap))
    return result


def nearest_categories(category: str) -> list[str]:
    """All other GoEmotions categories, sorted by VA distance from `category`
    (closest first). Used to sample 'VA-close different-category' vs.
    'VA-far different-category' partners for a given anchor's category."""
    others = [c for c in GOEMOTIONS_CATEGORIES if c != category]
    return sorted(others, key=lambda c: euclidean_distance(VA_LOOKUP[category], VA_LOOKUP[c]))


def _sample_up_to(pool: list[dict], k: int, rng: random.Random) -> list[dict]:
    return rng.sample(pool, min(k, len(pool)))


def sample_partners(
    anchor: dict,
    grouped: dict[str, list[dict]],
    k_same: int = 3,
    k_diff_close: int = 2,
    k_diff_far: int = 2,
    seed: int = 401,
) -> list[dict]:
    """Sample partner examples for one anchor: k_same from the anchor's own
    primary-category group (excluding the anchor), k_diff_close from the
    nearest other categories by VA distance, k_diff_far from the farthest --
    deliberately includes the VA-close-different-category partners (the
    anger/fear-style cap-triggering pairs) rather than leaving them to
    chance under pure random sampling."""
    rng = random.Random(seed)
    primary = anchor["labels"][0]

    same_pool = [e for e in grouped.get(primary, []) if e["id"] != anchor["id"]]
    same_partners = _sample_up_to(same_pool, k_same, rng)

    ranking = nearest_categories(primary)
    half = len(ranking) // 2
    close_categories, far_categories = ranking[:half], ranking[half:]

    close_pool = [e for cat in close_categories for e in grouped.get(cat, [])]
    far_pool = [e for cat in far_categories for e in grouped.get(cat, [])]
    close_partners = _sample_up_to(close_pool, k_diff_close, rng)
    far_partners = _sample_up_to(far_pool, k_diff_far, rng)

    return same_partners + close_partners + far_partners


def generate_training_pairs(
    examples: list[dict],
    cap: int = 1500,
    k_same: int = 3,
    k_diff_close: int = 2,
    k_diff_far: int = 2,
    anchor_seed: int = 402,
    partner_seed_base: int = 403,
) -> list[dict]:
    """Top-level: cap anchors per category (so common categories like neutral
    don't dominate and starve rare ones like grief of training signal), sample
    partners for each anchor (same-category + VA-close-different +
    VA-far-different), and compute the exact target similarity for each
    resulting pair via compute_target_similarity. Returns pairs ready for
    CosineSimilarityLoss training: {"text_a", "text_b", "similarity"}.
    Each example needs "text" and "labels" (list[str]) keys."""
    grouped = group_by_primary_category(examples)
    anchors = cap_per_category(grouped, cap=cap, seed=anchor_seed)

    pairs = []
    for i, anchor in enumerate(anchors):
        partners = sample_partners(
            anchor, grouped, k_same=k_same, k_diff_close=k_diff_close, k_diff_far=k_diff_far,
            seed=partner_seed_base + i,
        )
        for partner in partners:
            similarity = compute_target_similarity(anchor["labels"], partner["labels"])
            pairs.append({"text_a": anchor["text"], "text_b": partner["text"], "similarity": similarity})
    return pairs


def get_va_coordinates(labels: list[str]) -> tuple[float, float]:
    """VA point for an example: the lookup value for a single label, or the
    simple average across labels for a multi-label example (~17% of
    GoEmotions rows carry 2+ labels)."""
    if not labels:
        raise ValueError("labels must be non-empty")

    valences = [VA_LOOKUP[label][0] for label in labels]
    arousals = [VA_LOOKUP[label][1] for label in labels]
    return (sum(valences) / len(labels), sum(arousals) / len(labels))
