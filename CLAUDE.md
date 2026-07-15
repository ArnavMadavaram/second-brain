# CLAUDE.md — Second Brain

This file is the project context for Claude Code. Read it fully before writing code.

## What we are building

A local-first personal knowledge retrieval system ("Second Brain") that ingests the owner's own writing (a ChatGPT conversation export first, journal entries later) and answers natural-language questions about his own life using his own words.

The distinctive research contribution is a custom embedding model fine-tuned for emotional similarity, using sentence-transformers on the GoEmotions dataset, evaluated against a standard off-the-shelf baseline. The emotional framework is anchored in Russell's circumplex model of affect (valence and arousal) and Lerner's Appraisal Tendency Framework. The value of the project lives in the research question, the contrastive fine-tuning method, and the evaluation rigor. Model size is not the point.

This is a CS494 directed study (six-week course, per the submitted proposal). Owner is a CS student. He thinks independently, works step by step, and wants honest pushback, not validation. Flag scope creep, circular reasoning, and weak decisions directly.

## Six week plan (from the submitted proposal)

- Week 1: Finalize proposal and design. Set up the project, version control, and the basic data ingestion for the ChatGPT export.
- Week 2: Build the classifier that separates personal from task content. Get basic retrieval working with a standard embedding model as the baseline.
- Week 3: Fine-tune the custom emotion embedding model on public data. Begin comparing it against the baseline.
- Week 4: Add emotional tagging at ingestion. Build the daily check-in and streak feature. Build the evaluation framework. Begin documentation.
- Week 5: Run the full evaluation, custom model versus baseline. If retrieval is working well, attempt the decision prediction prototype. Refine based on results.
- Week 6: Final testing, finish documentation, write the final report, and prepare the prototype demonstration.

## Core values (these drive architecture)

- Local-first. No cloud for core personal data.
- Avoid LLM circularity. Do not use an LLM's output as emotional ground truth for a system that will later feed an LLM. This is why emotion detection uses a pretrained GoEmotions classifier, not an LLM call.
- Classifier and embedder are distinct components with distinct jobs. Do not conflate them.
- Parsing is lossless. Every "should this count" judgment is deferred to a later, reversible stage. You can always discard data later; you can never recover it.

## Environment

- macOS, MacBook Pro M2, MPS available.
- Python 3.12 via Homebrew. Virtual environment at `.venv`.
- Installed stack: `sentence-transformers[train]`, `chromadb`, `fastapi`, `uvicorn[standard]`, `anthropic`, `python-dotenv`. Pinned in `pyproject.toml`; full transitive freeze in `requirements-lock.txt`.
- Storage: ChromaDB (vector), SQLite (structured app data), JSONL (intermediate parsing output).
- Training: MPS locally for development, Google Colab free tier for fine-tuning runs. Training code is written device-agnostic (auto device detection, configurable paths).
- Versioning: Git and GitHub for code. No Claude Code co-author signature on commits. OneDrive holds only a static backup of the raw export.
- The project lives locally, OUTSIDE OneDrive. Never read live databases or working files from OneDrive; its sync and file dehydration corrupt SQLite and ChromaDB.
- Raw export is copied into `data/raw/` locally (gitignored). Do not read the shards from OneDrive.
- Base model for later fine-tuning: `all-MiniLM-L6-v2`.

## The four gates (locked architecture)

Data passes through four gates. "Drop ChatGPT responses" resolves differently at each. Do not collapse them.

1. **Parse / store.** Keep everything, both roles, lossless, with `role` and `on_active_path` as tags rather than filters. This is the parser's job (below).
2. **Classify (emotion labels).** User turns only. Assistant text never receives an emotion label. Two reasons: ChatGPT's RLHF positivity would skew the affect distribution, and labeling its text as the owner's emotion is the LLM-in-LLM-out circularity the project rejects. Also note assistant turns often mirror the user's stated feelings, which would inflate apparent emotional density and poison contrastive training pairs.
3. **Embed (retrievable chunks).** Both roles embedded, with `role` in ChromaDB metadata. Reason: there are two retrieval modes. Emotional retrieval is user-only by construction (assistant chunks have no valid emotion label from gate 2). Factual retrieval wants assistant text. One store, two filtered views. Assistant text is roughly 2.2x user volume, so user-only is the DEFAULT filter and assistant chunks are opt-in per query, or default search silently returns mostly-ChatGPT. This gate is decided but is the one place to revisit once real retrieval is observable.
4. **Answer-time context.** Keep assistant text. When a user chunk retrieves, the adjacent assistant turn rides along into the LLM context so "what did I decide about X" can surface the actual explanation.

Note: role filtering and personal/task filtering are ORTHOGONAL. The owner's own messages are mostly task content (chemistry, calculus, coding homework). Dropping assistant turns removes zero task junk from the emotional pool. A separate personal/task classifier operates on user messages. Do not merge these two filters.

## ChatGPT export structure (verified against real shards)

- Input is seven files at `data/raw/chatGPTExport/conversations-000.json` through `conversations-006.json` (three-digit, glob as `conversations-*.json`), each a JSON array of conversation objects. That directory also contains `conversation_asset_file_names.json`, `export_manifest.json`, `library_files.json`, `shared_conversations.json`, `user.json`, `user_settings.json` — none of those are conversation shards, the glob above excludes them cleanly.
- Each conversation has `mapping` (node_id to node), `current_node`, `title`, `create_time`, `conversation_id`, `id`.
- Each node in `mapping` is `{id, message, parent, children}`.
- `message` can be null. That is the synthetic tree root, one per conversation. Skip these.
- `message.author.role`: in the oldest shard only `user` and `assistant` appear. Newer shards very likely add `tool` and possibly `system`.
- `message.content.content_type`: oldest shard has `text` and `multimodal_text`. Newer shards likely add `code` and others.
- `message.content.parts` is a heterogeneous list: plain strings for `text`, dicts with `asset_pointer` for `multimodal_text` (image uploads).
- The conversation "as it happened" is the path from `current_node` up the `parent` chain, reversed. Branches exist from prompt edits and regenerations. The oldest shard had about 4% of message nodes off the active path.
- IMPORTANT: the shard used to verify this is 2023 data, from before code interpreter, browsing, custom GPTs, and memory. Do not assume the roles and content types seen there are the full set. The census (below) exists precisely to surface what the newer shards contain.

## The parser (Week 1 deliverable)

Output format: JSONL, one object per message node, at `data/processed/messages.jsonl` (gitignored). Not SQLite. Loading JSONL into SQLite is a separate later script; keep the stages decoupled.

One record per message node with these fields:

- `conversation_id`
- `conversation_title`
- `message_id`
- `parent_id`
- `role`
- `create_time`
- `content_type`
- `text`
- `on_active_path` (bool)
- `has_attachment` (bool)

Rules:

- Glob all shards in `data/raw/chatGPTExport/`, load each, process all conversations.
- Reconstruct the active path per conversation by walking `current_node` up the `parent` chain and reversing. Tag each emitted node with `on_active_path` true or false. Do NOT drop off-path nodes; tagging keeps the discard reversible downstream.
- Emit every message node, both roles. Lossless.
- Skip null-message nodes (synthetic roots). This is the only exception to lossless, because they carry no content, only a parent pointer already used.
- Build `text` by joining only the string elements of `content.parts`. If any non-string part is present, set `has_attachment` true and ignore the binary (the `.dat` assets are out of scope).
- Be defensive. An unknown `role` or `content_type` must be recorded, not crashed on and not silently dropped.
- At the end, print a CENSUS: counts of every `role` and every `content_type` seen across all shards, plus counts of null-message nodes, off-path nodes, and records with attachments. The census is the real deliverable of this step. It is the first real look at what the newer shards contain.

## Week 2 deliverables (scoped down for a hard deadline — see plan)

- **Personal/task classifier**: separates emotionally-relevant personal content from task content, operating on user turns only. This is orthogonal to the role filter above, not a substitute for it. First pass: ~60-100 hand-labeled held-out examples, zero-shot NLI classifier, evaluated with precision/recall/F1 and a confusion matrix. Flagged explicitly as a fast v1, not the final rigorous version — revisit with a larger labeled set later if time allows.
- **Baseline retrieval**: `all-MiniLM-L6-v2` embeddings into ChromaDB, both roles, `role` and `is_personal` as separate orthogonal metadata fields. Message-level chunking to start — this choice must be reused unchanged later when the custom model is swapped in, so the comparison stays fair.
- GoEmotions inference (emotion tagging) is deferred to Week 4 per the proposal's own plan — not attempted in Week 2.

## Working style

- Deliberate, step by step. Do not race ahead to the next component.
- Plain language. No em dashes.
- Honest pushback over agreement. If something in this brief looks wrong when checked against the real data, say so rather than complying silently.
- Pair-programming, TDD-style for logic-bearing code (tree walking, text extraction, classifier evaluation). Direct implementation for glue/boilerplate is fine.
