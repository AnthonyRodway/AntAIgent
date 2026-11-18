from antaigent.chats import ChatStore, SavedTurn, chats_path, make_title
from antaigent.projects import index_path


def test_chats_are_saved_listed_and_deleted(tmp_path):
    with ChatStore(tmp_path / "chats.db") as chats:
        first = chats.create("How does learning work?")
        chats.add_turn(first, SavedTurn(
            "How does learning work?", "It walks the folder.", "thinking...",
            sources=[("src/indexer.py", 10, 40)], graph=["learn"], summaries=["src/"],
            stats={"eval_count": 120, "eval_duration": 1_500_000_000, "message": {"role": "assistant"}},
        ))
        second = chats.create("Second")
        chats.add_turn(second, SavedTurn("Second", error="Can't reach Ollama", stopped=True))
        chats.add_turn(first, SavedTurn("And then?", "It embeds the chunks."))

        listed = chats.conversations()
        assert [c.id for c in listed] == [first, second]  # most recently used first
        assert [c.turns for c in listed] == [2, 1]

        turns = chats.turns(first)
        assert [t.question for t in turns] == ["How does learning work?", "And then?"]
        assert turns[0].sources == [("src/indexer.py", 10, 40)]
        assert (turns[0].graph, turns[0].summaries) == (["learn"], ["src/"])
        assert turns[0].stats == {"eval_count": 120, "eval_duration": 1_500_000_000}
        assert chats.turns(second)[0].stopped and chats.turns(second)[0].error

        chats.rename(first, "Learning")
        chats.delete(second)
        assert [(c.title, c.turns) for c in chats.conversations()] == [("Learning", 2)]
        assert chats.turns(second) == []


def test_chats_live_beside_the_index_not_inside_it(tmp_path):
    root = tmp_path / "project"
    assert chats_path(tmp_path, root).parent == index_path(tmp_path, root).parent
    assert chats_path(tmp_path, root).name == "chats.db"


def test_titles_are_one_short_line():
    assert make_title("  what   does\nthis do?  ") == "what does this do?"
    long = make_title("word " * 40)
    assert len(long) == 60 and long.endswith("...")
