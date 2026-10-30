from geoqa.kb import Entry, KnowledgeBase


def make_kb():
    raw = [
        {"term": "红壤", "category": "SOILVEG", "aliases": ["红色土壤"], "is_a": ["土壤类型"],
         "features": ["酸性", "有机质含量低", "铁铝氧化物富集"], "causes": ["高温多雨", "淋溶作用强"],
         "locations": ["南方低山丘陵"]},
        {"term": "黑土", "category": "SOILVEG", "is_a": ["土壤类型"], "features": ["有机质含量高", "土层深厚"],
         "causes": ["冬季寒冷", "有机质分解慢"], "locations": ["东北平原"]},
        {"term": "淋溶作用", "category": "SOILVEG", "is_a": ["成土过程"], "causes": ["降水丰富"],
         "effects": ["盐基离子流失", "土壤酸化"]},
        {"term": "x", "category": "SOILVEG"},  # too short -> rejected
        {"term": "荒漠土", "category": "土壤与植被", "features": "有机质含量低、土层薄"},
    ]
    entries = [e for e in (Entry.from_llm(r, ["test"]) for r in raw) if e]
    return KnowledgeBase(entries)


def test_entry_validation():
    kb = make_kb()
    assert set(kb.entries) == {"红壤", "黑土", "淋溶作用", "荒漠土"}
    assert kb.entries["荒漠土"].category == "SOILVEG"  # Chinese category name mapped to id
    assert kb.entries["荒漠土"].features == ["有机质含量低", "土层薄"]  # string split into phrases


def test_alias_and_neighbors():
    kb = make_kb()
    assert kb.canon("红色土壤") == "红壤"
    assert "酸性" in kb.neighbors("红壤")
    assert "淋溶作用强" in kb.neighbors("红壤")
    assert "淋溶作用" in kb.neighbors("红壤")  # contained in the phrase "淋溶作用强"
    assert kb.degree["土壤类型"] == 2
    assert kb.idf("土壤类型") < kb.idf("酸性")


def test_question_local_copy_does_not_mutate_base():
    kb = make_kb()
    extra = Entry.from_llm({"term": "砖红壤", "is_a": ["红壤"], "features": ["酸性"]}, ["dcr"])
    local = kb.with_entries([extra])
    assert "砖红壤" in local.entries and "砖红壤" not in kb.entries
    assert local.degree["红壤"] == 1 and kb.degree.get("红壤", 0) == 0
    assert local.canon("砖红壤") == "砖红壤" and "砖红壤" not in kb.surface
    assert "红壤" in local.neighbors("砖红壤")
    assert local.get("黑土") is kb.get("黑土")
