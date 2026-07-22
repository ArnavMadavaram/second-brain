import math

import pytest

from second_brain.goemotions_pairs import (
    GOEMOTIONS_CATEGORIES,
    VA_LOOKUP,
    base_similarity_from_distance,
    build_evaluation_triplets,
    cap_per_category,
    compute_target_similarity,
    euclidean_distance,
    get_va_coordinates,
    generate_training_pairs,
    group_by_primary_category,
    nearest_categories,
    sample_partners,
    shares_category,
)


def test_va_lookup_covers_all_27_categories_plus_neutral():
    assert len(GOEMOTIONS_CATEGORIES) == 28  # 27 emotions + neutral
    for category in GOEMOTIONS_CATEGORIES:
        assert category in VA_LOOKUP


def test_va_lookup_known_values_match_nrc_vad_lexicon():
    # spot-check against the actual downloaded lexicon values, not invented numbers
    assert VA_LOOKUP["joy"] == (0.980, 0.824)
    assert VA_LOOKUP["sadness"] == (0.052, 0.288)
    assert VA_LOOKUP["anger"] == (0.167, 0.865)
    assert VA_LOOKUP["fear"] == (0.073, 0.840)
    assert VA_LOOKUP["neutral"] == (0.469, 0.184)


def test_get_va_coordinates_single_label_returns_lookup_value():
    coords = get_va_coordinates(["joy"])
    assert coords == VA_LOOKUP["joy"]


def test_get_va_coordinates_multi_label_averages():
    # two labels -> simple average of their VA points
    coords = get_va_coordinates(["joy", "sadness"])
    expected_v = (VA_LOOKUP["joy"][0] + VA_LOOKUP["sadness"][0]) / 2
    expected_a = (VA_LOOKUP["joy"][1] + VA_LOOKUP["sadness"][1]) / 2
    assert coords == (expected_v, expected_a)


def test_get_va_coordinates_raises_on_empty_labels():
    with pytest.raises(ValueError):
        get_va_coordinates([])


def test_euclidean_distance_zero_for_identical_points():
    assert euclidean_distance((0.5, 0.5), (0.5, 0.5)) == 0.0


def test_euclidean_distance_anger_fear_matches_worked_example():
    # from the plan: anger (.167,.865) vs fear (.073,.840) -> distance ~0.097
    d = euclidean_distance(VA_LOOKUP["anger"], VA_LOOKUP["fear"])
    assert d == pytest.approx(0.0973, abs=0.001)


def test_euclidean_distance_max_possible_is_sqrt_2():
    d = euclidean_distance((0.0, 0.0), (1.0, 1.0))
    assert d == pytest.approx(math.sqrt(2))


def test_base_similarity_is_one_at_zero_distance():
    assert base_similarity_from_distance(0.0) == pytest.approx(1.0)


def test_base_similarity_is_zero_at_max_distance():
    assert base_similarity_from_distance(math.sqrt(2)) == pytest.approx(0.0)


def test_base_similarity_anger_fear_is_high():
    d = euclidean_distance(VA_LOOKUP["anger"], VA_LOOKUP["fear"])
    sim = base_similarity_from_distance(d)
    assert sim == pytest.approx(0.931, abs=0.005)


def test_base_similarity_joy_sadness_is_low():
    d = euclidean_distance(VA_LOOKUP["joy"], VA_LOOKUP["sadness"])
    sim = base_similarity_from_distance(d)
    assert sim == pytest.approx(0.242, abs=0.005)


def test_shares_category_true_for_identical_single_labels():
    assert shares_category(["joy"], ["joy"]) is True


def test_shares_category_false_for_disjoint_labels():
    assert shares_category(["anger"], ["fear"]) is False


def test_shares_category_true_for_multi_label_overlap():
    # share "love" even though the rest of each label set differs
    assert shares_category(["joy", "love"], ["love", "pride"]) is True


def test_shares_category_false_for_multi_label_no_overlap():
    assert shares_category(["joy", "love"], ["anger", "fear"]) is False


def test_compute_target_similarity_same_category_not_capped():
    # same single label -> distance 0 -> base similarity 1.0, same category
    # means the cap never applies regardless of its value
    sim = compute_target_similarity(["joy"], ["joy"], cap=0.7)
    assert sim == pytest.approx(1.0)


def test_compute_target_similarity_different_category_gets_capped():
    # anger/fear raw similarity (~0.931) exceeds the cap -> capped down
    sim = compute_target_similarity(["anger"], ["fear"], cap=0.7)
    assert sim == pytest.approx(0.7)


def test_compute_target_similarity_different_category_below_cap_untouched():
    # joy/sadness raw similarity (~0.242) is already below the cap -> unchanged,
    # the cap must never artificially RAISE a similarity
    sim = compute_target_similarity(["joy"], ["sadness"], cap=0.7)
    assert sim == pytest.approx(0.242, abs=0.005)


def test_compute_target_similarity_default_cap_is_point_seven():
    capped = compute_target_similarity(["anger"], ["fear"])
    assert capped == pytest.approx(0.7)


def test_compute_target_similarity_multi_label_shared_category_not_capped():
    # share "love" -> treated as same category -> no cap even though these
    # labels individually might otherwise trigger one
    sim = compute_target_similarity(["joy", "love"], ["love", "grief"], cap=0.3)
    uncapped = base_similarity_from_distance(
        euclidean_distance(get_va_coordinates(["joy", "love"]), get_va_coordinates(["love", "grief"]))
    )
    assert sim == pytest.approx(uncapped)


def _ex(id_, labels, text=""):
    return {"id": id_, "labels": labels, "text": text or id_}


def test_group_by_primary_category_uses_first_label():
    examples = [
        _ex("a", ["joy"]),
        _ex("b", ["joy", "love"]),  # primary is joy even though multi-label
        _ex("c", ["anger"]),
    ]
    grouped = group_by_primary_category(examples)
    assert {e["id"] for e in grouped["joy"]} == {"a", "b"}
    assert {e["id"] for e in grouped["anger"]} == {"c"}


def test_group_by_primary_category_omits_empty_groups():
    examples = [_ex("a", ["joy"])]
    grouped = group_by_primary_category(examples)
    assert "anger" not in grouped  # no anger examples given, no empty-list entry


def test_cap_per_category_keeps_all_examples_below_cap():
    grouped = {"grief": [_ex(f"g{i}", ["grief"]) for i in range(5)]}
    result = cap_per_category(grouped, cap=1500, seed=1)
    assert len(result) == 5


def test_cap_per_category_downsamples_above_cap():
    grouped = {"neutral": [_ex(f"n{i}", ["neutral"]) for i in range(100)]}
    result = cap_per_category(grouped, cap=10, seed=1)
    assert len(result) == 10
    # must be an actual subset, not fabricated examples
    result_ids = {e["id"] for e in result}
    original_ids = {e["id"] for e in grouped["neutral"]}
    assert result_ids <= original_ids


def test_cap_per_category_is_reproducible_with_same_seed():
    grouped = {"neutral": [_ex(f"n{i}", ["neutral"]) for i in range(100)]}
    first = cap_per_category(grouped, cap=10, seed=5)
    second = cap_per_category(grouped, cap=10, seed=5)
    assert [e["id"] for e in first] == [e["id"] for e in second]


def test_cap_per_category_combines_across_categories():
    grouped = {
        "neutral": [_ex(f"n{i}", ["neutral"]) for i in range(100)],
        "grief": [_ex(f"g{i}", ["grief"]) for i in range(5)],
    }
    result = cap_per_category(grouped, cap=10, seed=1)
    assert len(result) == 15  # 10 capped from neutral + all 5 from grief


def test_nearest_categories_excludes_the_category_itself():
    result = nearest_categories("anger")
    assert "anger" not in result


def test_nearest_categories_includes_every_other_category():
    result = nearest_categories("anger")
    assert len(result) == len(GOEMOTIONS_CATEGORIES) - 1
    assert set(result) == set(GOEMOTIONS_CATEGORIES) - {"anger"}


def test_nearest_categories_sorted_closest_first():
    result = nearest_categories("anger")
    # fear should rank near the top -- it's the worked example of VA-closeness
    assert result.index("fear") < len(result) // 3
    # sadness (low arousal, opposite of anger's high arousal) should rank far
    assert result.index("fear") < result.index("sadness")


def _full_grouped(n_per_category=4):
    return group_by_primary_category(
        [_ex(f"{cat}-{i}", [cat]) for cat in GOEMOTIONS_CATEGORIES for i in range(n_per_category)]
    )


def test_sample_partners_excludes_the_anchor_itself():
    grouped = _full_grouped()
    anchor = grouped["anger"][0]
    partners = sample_partners(anchor, grouped, k_same=3, k_diff_close=2, k_diff_far=2, seed=1)
    assert anchor["id"] not in {p["id"] for p in partners}


def test_sample_partners_returns_requested_counts_when_available():
    grouped = _full_grouped(n_per_category=10)  # plenty of same-category partners available
    anchor = grouped["anger"][0]
    partners = sample_partners(anchor, grouped, k_same=3, k_diff_close=2, k_diff_far=2, seed=1)
    assert len(partners) == 7  # 3 + 2 + 2


def test_sample_partners_same_category_partners_share_anchor_category():
    grouped = _full_grouped()
    anchor = grouped["anger"][0]
    partners = sample_partners(anchor, grouped, k_same=3, k_diff_close=0, k_diff_far=0, seed=1)
    assert all(p["labels"][0] == "anger" for p in partners)


def test_sample_partners_close_partners_are_va_closer_than_far_partners():
    grouped = _full_grouped()
    anchor = grouped["anger"][0]
    close = sample_partners(anchor, grouped, k_same=0, k_diff_close=5, k_diff_far=0, seed=1)
    far = sample_partners(anchor, grouped, k_same=0, k_diff_close=0, k_diff_far=5, seed=1)

    ranking = nearest_categories("anger")
    close_ranks = [ranking.index(p["labels"][0]) for p in close]
    far_ranks = [ranking.index(p["labels"][0]) for p in far]
    assert max(close_ranks) < min(far_ranks)  # every close pick ranks nearer than every far pick


def test_sample_partners_is_reproducible_with_same_seed():
    grouped = _full_grouped()
    anchor = grouped["anger"][0]
    first = sample_partners(anchor, grouped, seed=7)
    second = sample_partners(anchor, grouped, seed=7)
    assert [p["id"] for p in first] == [p["id"] for p in second]


def test_sample_partners_caps_gracefully_when_pool_smaller_than_requested():
    grouped = _full_grouped(n_per_category=1)  # only 1 example per category available
    anchor = grouped["anger"][0]
    # k_same=3 requested but anger has only the anchor itself (0 other same-category examples)
    partners = sample_partners(anchor, grouped, k_same=3, k_diff_close=2, k_diff_far=2, seed=1)
    same_category_partners = [p for p in partners if p["labels"][0] == "anger"]
    assert len(same_category_partners) == 0  # doesn't crash, just returns what's available


def test_generate_training_pairs_produces_expected_count():
    examples = [_ex(f"{cat}-{i}", [cat], text=f"{cat} text {i}") for cat in GOEMOTIONS_CATEGORIES for i in range(10)]
    pairs = generate_training_pairs(examples, cap=1500, k_same=3, k_diff_close=2, k_diff_far=2)
    # 28 categories x 10 examples each = 280 anchors (none capped, cap=1500), 7 partners each
    assert len(pairs) == 280 * 7


def test_generate_training_pairs_similarity_matches_compute_target_similarity():
    examples = [_ex(f"{cat}-{i}", [cat], text=f"{cat} text {i}") for cat in GOEMOTIONS_CATEGORIES for i in range(10)]
    pairs = generate_training_pairs(examples, cap=1500, k_same=1, k_diff_close=1, k_diff_far=1)
    # every generated pair's similarity should be independently reproducible
    # by calling compute_target_similarity on the same two categories
    anger_pairs = [p for p in pairs if "anger" in p["text_a"]]
    for p in anger_pairs:
        other_category = p["text_b"].split(" text")[0]
        expected = compute_target_similarity(["anger"], [other_category])
        assert p["similarity"] == pytest.approx(expected)


def test_generate_training_pairs_respects_category_cap():
    examples = [_ex(f"neutral-{i}", ["neutral"], text=f"n{i}") for i in range(100)]
    examples += [_ex(f"grief-{i}", ["grief"], text=f"g{i}") for i in range(5)]
    pairs = generate_training_pairs(examples, cap=10, k_same=0, k_diff_close=0, k_diff_far=0)
    # 10 neutral anchors (capped from 100) + 5 grief anchors (below cap), 0 partners each -> 0 pairs
    # but we can check the anchor count indirectly: with k=0 there should be no pairs at all,
    # so instead verify via a version that keeps same-category partners
    pairs_with_partners = generate_training_pairs(examples, cap=10, k_same=3, k_diff_close=0, k_diff_far=0)
    neutral_anchor_pairs = [p for p in pairs_with_partners if "n" in p["text_a"] and p["text_a"].startswith("n")]
    # at most 10 neutral anchors x 3 same-category partners
    assert len(neutral_anchor_pairs) <= 10 * 3


def test_build_evaluation_triplets_returns_requested_structure():
    examples = [_ex(f"{cat}-{i}", [cat], text=f"{cat} text {i}") for cat in GOEMOTIONS_CATEGORIES for i in range(5)]
    triplets = build_evaluation_triplets(examples, n_anchors=20, seed=1)
    assert len(triplets) > 0
    for t in triplets:
        assert set(t.keys()) == {"anchor", "positive", "negative_close", "negative_far"}


def test_build_evaluation_triplets_caps_at_n_anchors():
    examples = [_ex(f"{cat}-{i}", [cat], text=f"{cat} text {i}") for cat in GOEMOTIONS_CATEGORIES for i in range(5)]
    triplets = build_evaluation_triplets(examples, n_anchors=10, seed=1)
    assert len(triplets) <= 10


def test_build_evaluation_triplets_skips_anchors_without_enough_partners():
    # only 1 category with any real pool -- no different-category partners exist at all
    examples = [_ex(f"joy-{i}", ["joy"], text=f"joy text {i}") for i in range(5)]
    triplets = build_evaluation_triplets(examples, n_anchors=10, seed=1)
    assert triplets == []  # no close/far partners possible -- must not crash, just skip


def test_build_evaluation_triplets_is_reproducible_with_same_seed():
    examples = [_ex(f"{cat}-{i}", [cat], text=f"{cat} text {i}") for cat in GOEMOTIONS_CATEGORIES for i in range(5)]
    first = build_evaluation_triplets(examples, n_anchors=15, seed=3)
    second = build_evaluation_triplets(examples, n_anchors=15, seed=3)
    assert first == second
