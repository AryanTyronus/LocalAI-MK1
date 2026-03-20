from memory.fact_memory import FactMemory


class _FakeModel:
    def embed(self, texts):
        return [[float(len(t) % 7), 1.0, 0.5] for t in texts]


def test_fact_extraction_and_dedup(tmp_path):
    facts_path = tmp_path / "facts.json"
    fm = FactMemory(model_manager=_FakeModel(), filepath=str(facts_path), similarity_threshold=0.99)
    extracted = fm.extract("My name is Aryan. I like Python programming.")
    assert extracted
    added_1 = fm.add_facts(extracted)
    added_2 = fm.add_facts(extracted)
    assert added_1 >= 1
    assert added_2 == 0
