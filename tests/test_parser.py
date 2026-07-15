from second_brain.parser import build_census, extract_text_and_attachment, parse_conversation


def _conversation_with_null_root():
    return {
        "conversation_id": "conv-1",
        "title": "Test Conversation",
        "current_node": "node-1",
        "mapping": {
            "root": {
                "id": "root",
                "message": None,
                "parent": None,
                "children": ["node-1"],
            },
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "node-1",
                    "author": {"role": "user"},
                    "create_time": 1000.0,
                    "content": {"content_type": "text", "parts": ["hello"]},
                },
                "parent": "root",
                "children": [],
            },
        },
    }


def test_parse_conversation_skips_null_message_root():
    records = parse_conversation(_conversation_with_null_root())

    message_ids = [r["message_id"] for r in records]
    assert "root" not in message_ids
    assert message_ids == ["node-1"]


def _message(node_id, role="user", text="hi", create_time=1000.0, content_type="text"):
    return {
        "id": node_id,
        "author": {"role": role},
        "create_time": create_time,
        "content": {"content_type": content_type, "parts": [text]},
    }


def _conversation_with_branch():
    # root -> node-1 -> node-2 (active, current_node) and node-2b (sibling, off-path)
    return {
        "conversation_id": "conv-branch",
        "title": "Branch Test",
        "current_node": "node-2",
        "mapping": {
            "root": {"id": "root", "message": None, "parent": None, "children": ["node-1"]},
            "node-1": {
                "id": "node-1",
                "message": _message("node-1"),
                "parent": "root",
                "children": ["node-2", "node-2b"],
            },
            "node-2": {
                "id": "node-2",
                "message": _message("node-2", role="assistant"),
                "parent": "node-1",
                "children": [],
            },
            "node-2b": {
                "id": "node-2b",
                "message": _message("node-2b", role="assistant", text="regenerated"),
                "parent": "node-1",
                "children": [],
            },
        },
    }


def test_parse_conversation_tags_off_active_path_nodes_without_dropping_them():
    records = parse_conversation(_conversation_with_branch())
    by_id = {r["message_id"]: r for r in records}

    # all three real messages are still present -- lossless
    assert set(by_id.keys()) == {"node-1", "node-2", "node-2b"}

    assert by_id["node-1"]["on_active_path"] is True
    assert by_id["node-2"]["on_active_path"] is True
    assert by_id["node-2b"]["on_active_path"] is False


def test_joins_multiple_string_parts_with_no_attachment():
    content = {"content_type": "text", "parts": ["hello ", "world"]}

    text, has_attachment = extract_text_and_attachment(content)

    assert text == "hello world"
    assert has_attachment is False


def test_non_string_part_sets_attachment_flag_and_is_excluded_from_text():
    content = {
        "content_type": "multimodal_text",
        "parts": ["a caption", {"asset_pointer": "file-service://abc", "content_type": "image_asset_pointer"}],
    }

    text, has_attachment = extract_text_and_attachment(content)

    assert text == "a caption"
    assert has_attachment is True


def test_missing_parts_key_returns_empty_text_no_attachment():
    content = {"content_type": "text"}

    text, has_attachment = extract_text_and_attachment(content)

    assert text == ""
    assert has_attachment is False


def test_reasoning_recap_extracts_top_level_content_string():
    content = {"content_type": "reasoning_recap", "content": "Thought for 6 seconds"}

    text, has_attachment = extract_text_and_attachment(content)

    assert text == "Thought for 6 seconds"
    assert has_attachment is False


def test_thoughts_joins_content_field_of_each_thought_entry():
    content = {
        "content_type": "thoughts",
        "source_analysis_msg_id": "abc",
        "thoughts": [
            {"content": "First reasoning step.", "summary": "Step one", "finished": True, "chunks": []},
            {"content": "Second reasoning step.", "summary": "Step two", "finished": True, "chunks": []},
        ],
    }

    text, has_attachment = extract_text_and_attachment(content)

    assert text == "First reasoning step.\nSecond reasoning step."
    assert has_attachment is False


def test_thoughts_with_empty_list_returns_empty_text_no_attachment():
    content = {"content_type": "thoughts", "source_analysis_msg_id": "abc", "thoughts": []}

    text, has_attachment = extract_text_and_attachment(content)

    assert text == ""
    assert has_attachment is False


def test_parse_conversation_builds_full_record_with_all_fields():
    conversation = {
        "conversation_id": "conv-full",
        "title": "Full Record Test",
        "current_node": "node-1",
        "mapping": {
            "root": {"id": "root", "message": None, "parent": None, "children": ["node-1"]},
            "node-1": {
                "id": "node-1",
                "message": _message("node-1", role="user", text="hello there", create_time=1700000000.0),
                "parent": "root",
                "children": [],
            },
        },
    }

    records = parse_conversation(conversation)

    assert len(records) == 1
    record = records[0]
    assert record == {
        "conversation_id": "conv-full",
        "conversation_title": "Full Record Test",
        "message_id": "node-1",
        "parent_id": "root",
        "role": "user",
        "create_time": 1700000000.0,
        "content_type": "text",
        "text": "hello there",
        "on_active_path": True,
        "has_attachment": False,
    }


def test_parse_conversation_is_defensive_on_unknown_role_and_missing_content():
    conversation = {
        "conversation_id": "conv-weird",
        "title": "Weird Node",
        "current_node": "node-1",
        "mapping": {
            "root": {"id": "root", "message": None, "parent": None, "children": ["node-1"]},
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "node-1",
                    "author": {"role": "tool"},
                    "create_time": None,
                    # no "content" key at all -- simulates an unanticipated newer-shard shape
                },
                "parent": "root",
                "children": [],
            },
        },
    }

    # must not raise
    records = parse_conversation(conversation)

    assert len(records) == 1
    record = records[0]
    assert record["role"] == "tool"
    assert record["content_type"] is None
    assert record["text"] == ""
    assert record["has_attachment"] is False


def test_build_census_counts_roles_content_types_and_flags():
    records = [
        {"role": "user", "content_type": "text", "on_active_path": True, "has_attachment": False},
        {"role": "user", "content_type": "text", "on_active_path": False, "has_attachment": True},
        {"role": "assistant", "content_type": "code", "on_active_path": True, "has_attachment": False},
    ]

    census = build_census(records, null_message_count=2)

    assert census["role_counts"] == {"user": 2, "assistant": 1}
    assert census["content_type_counts"] == {"text": 2, "code": 1}
    assert census["null_message_count"] == 2
    assert census["off_path_count"] == 1
    assert census["attachment_count"] == 1
    assert census["total_records"] == 3
