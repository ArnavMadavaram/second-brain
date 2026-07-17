from second_brain.labeling import (
    assign_split,
    remaining_to_label,
    sample_additional_messages_for_labeling,
    sample_messages_for_labeling,
    stratified_split,
)


def _user_msg(message_id, conversation_id, text="hello"):
    return {"message_id": message_id, "conversation_id": conversation_id, "role": "user", "text": text}


def _records_across_conversations(n_conversations, msgs_per_conversation=1):
    records = []
    for c in range(n_conversations):
        for m in range(msgs_per_conversation):
            records.append(_user_msg(f"msg-{c}-{m}", f"conv-{c}", text=f"message {c}-{m}"))
    return records


def test_sample_returns_at_most_one_message_per_conversation():
    records = _records_across_conversations(n_conversations=5, msgs_per_conversation=4)

    sampled = sample_messages_for_labeling(records, n=5, seed=1)

    conversation_ids = [r["conversation_id"] for r in sampled]
    assert len(sampled) == 5
    assert len(set(conversation_ids)) == 5  # no conversation contributes twice


def test_sample_caps_at_available_conversations():
    records = _records_across_conversations(n_conversations=3, msgs_per_conversation=2)

    sampled = sample_messages_for_labeling(records, n=250, seed=1)

    assert len(sampled) == 3  # can't exceed distinct conversation count


def test_sample_excludes_non_user_role_and_empty_text():
    records = [
        _user_msg("m1", "c1", text="real content"),
        {"message_id": "m2", "conversation_id": "c2", "role": "assistant", "text": "assistant reply"},
        {"message_id": "m3", "conversation_id": "c3", "role": "user", "text": "   "},
    ]

    sampled = sample_messages_for_labeling(records, n=10, seed=1)

    assert [r["message_id"] for r in sampled] == ["m1"]


def test_sample_is_reproducible_with_same_seed():
    records = _records_across_conversations(n_conversations=20, msgs_per_conversation=3)

    first = sample_messages_for_labeling(records, n=10, seed=7)
    second = sample_messages_for_labeling(records, n=10, seed=7)

    assert [r["message_id"] for r in first] == [r["message_id"] for r in second]


def test_assign_split_produces_requested_sizes_with_no_overlap():
    sampled = [_user_msg(f"m{i}", f"c{i}") for i in range(10)]

    split = assign_split(sampled, test_size=6, seed=1)

    test_ids = {r["message_id"] for r in split if r["split"] == "test"}
    dev_ids = {r["message_id"] for r in split if r["split"] == "dev"}
    assert len(test_ids) == 6
    assert len(dev_ids) == 4
    assert test_ids.isdisjoint(dev_ids)
    assert test_ids | dev_ids == {r["message_id"] for r in sampled}


def test_remaining_to_label_excludes_already_labeled_messages():
    sample = [
        _user_msg("m1", "c1"),
        _user_msg("m2", "c2"),
        _user_msg("m3", "c3"),
    ]
    existing_labels = {"m2": {"message_id": "m2", "label": "task"}}

    remaining = remaining_to_label(sample, existing_labels)

    assert [r["message_id"] for r in remaining] == ["m1", "m3"]


def test_remaining_to_label_returns_everything_when_no_labels_exist():
    sample = [_user_msg("m1", "c1"), _user_msg("m2", "c2")]

    remaining = remaining_to_label(sample, existing_labels={})

    assert len(remaining) == 2


def _labeled(message_id, label):
    return {"message_id": message_id, "conversation_id": f"c-{message_id}", "label": label, "split": "test"}


def test_stratified_split_spreads_rare_label_across_all_three_splits():
    # 10 personal, 40 task -- rare class must still land in train/dev/test
    labels = [_labeled(f"p{i}", "personal") for i in range(10)] + [_labeled(f"t{i}", "task") for i in range(40)]

    split = stratified_split(labels, train_frac=0.6, dev_frac=0.2, seed=1)

    personal_splits = {r["split"] for r in split if r["label"] == "personal"}
    assert personal_splits == {"train", "dev", "test"}


def test_stratified_split_preserves_total_count_and_labels():
    labels = [_labeled(f"p{i}", "personal") for i in range(10)] + [_labeled(f"t{i}", "task") for i in range(40)]

    split = stratified_split(labels, train_frac=0.6, dev_frac=0.2, seed=1)

    assert len(split) == len(labels)
    original_labels = {r["message_id"]: r["label"] for r in labels}
    for r in split:
        assert r["label"] == original_labels[r["message_id"]]  # labels never change, only split


def test_stratified_split_approximates_requested_fractions_per_label():
    labels = [_labeled(f"t{i}", "task") for i in range(100)]

    split = stratified_split(labels, train_frac=0.6, dev_frac=0.2, seed=1)

    counts = {"train": 0, "dev": 0, "test": 0}
    for r in split:
        counts[r["split"]] += 1
    assert counts["train"] == 60
    assert counts["dev"] == 20
    assert counts["test"] == 20


def test_stratified_split_single_example_label_does_not_crash():
    labels = [_labeled("e1", "exclude")]

    split = stratified_split(labels, train_frac=0.6, dev_frac=0.2, seed=1)

    assert len(split) == 1


def test_stratified_split_is_reproducible_with_same_seed():
    labels = [_labeled(f"p{i}", "personal") for i in range(10)] + [_labeled(f"t{i}", "task") for i in range(40)]

    first = stratified_split(labels, seed=5)
    second = stratified_split(labels, seed=5)

    assert [r["split"] for r in first] == [r["split"] for r in second]


def test_sample_additional_excludes_given_message_ids():
    records = _records_across_conversations(n_conversations=5, msgs_per_conversation=3)
    exclude = {"msg-0-0", "msg-1-0"}

    sampled = sample_additional_messages_for_labeling(records, exclude_message_ids=exclude, n=100, seed=1)

    sampled_ids = {r["message_id"] for r in sampled}
    assert sampled_ids.isdisjoint(exclude)


def test_sample_additional_respects_max_per_conversation():
    # one conversation with far more messages than the cap
    records = _records_across_conversations(n_conversations=1, msgs_per_conversation=20)

    sampled = sample_additional_messages_for_labeling(
        records, exclude_message_ids=set(), n=100, max_per_conversation=5, seed=1
    )

    assert len(sampled) == 5  # capped, even though 20 were available and n=100 was requested


def test_sample_additional_spreads_before_repeating():
    # 10 conversations, 3 messages each -- requesting fewer than 10 should touch
    # many distinct conversations (round-robin) rather than draining the first ones
    records = _records_across_conversations(n_conversations=10, msgs_per_conversation=3)

    sampled = sample_additional_messages_for_labeling(
        records, exclude_message_ids=set(), n=10, max_per_conversation=3, seed=1
    )

    conversation_ids = {r["conversation_id"] for r in sampled}
    assert len(conversation_ids) == 10  # spread across all 10, not piled into 3-4


def test_sample_additional_caps_at_available_candidates():
    records = _records_across_conversations(n_conversations=3, msgs_per_conversation=2)

    sampled = sample_additional_messages_for_labeling(
        records, exclude_message_ids=set(), n=1000, max_per_conversation=5, seed=1
    )

    assert len(sampled) == 6  # only 6 candidate messages exist total


def test_sample_additional_is_reproducible_with_same_seed():
    records = _records_across_conversations(n_conversations=20, msgs_per_conversation=5)

    first = sample_additional_messages_for_labeling(records, exclude_message_ids=set(), n=30, seed=9)
    second = sample_additional_messages_for_labeling(records, exclude_message_ids=set(), n=30, seed=9)

    assert [r["message_id"] for r in first] == [r["message_id"] for r in second]
