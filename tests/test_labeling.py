from second_brain.labeling import assign_split, remaining_to_label, sample_messages_for_labeling


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
