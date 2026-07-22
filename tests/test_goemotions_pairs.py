import math

import pytest

from second_brain.goemotions_pairs import (
    GOEMOTIONS_CATEGORIES,
    VA_LOOKUP,
    base_similarity_from_distance,
    compute_target_similarity,
    euclidean_distance,
    get_va_coordinates,
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
