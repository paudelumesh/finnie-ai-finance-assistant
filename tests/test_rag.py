import pytest

from src.rag.chunker import chunk_article, chunk_articles
from src.rag.embeddings import HashEmbedder
from src.rag.loader import Article, load_articles, parse_article
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore


def make_article(slug, title, category, body):
    return Article(slug, title, category, "Investopedia", "https://www.investopedia.com", body)


ARTICLES = [
    make_article("compound", "Compound interest", "basics",
                 "## Idea\nCompound interest means earning interest on your interest over many years.\n"
                 "## Example\nA deposit grows faster when interest compounds annually."),
    make_article("etf", "ETFs explained", "funds",
                 "## What\nAn exchange traded fund holds a basket of stocks and trades on an exchange.\n"
                 "## Fees\nETFs usually have low expense ratios."),
    make_article("roth", "Roth IRA", "retirement",
                 "## Basics\nA Roth IRA is a retirement account funded with after tax money."),
]


@pytest.fixture
def store():
    return VectorStore.build(chunk_articles(ARTICLES, 400, 50), HashEmbedder())


def write_md(tmp_path, name, front, body="Body text"):
    p = tmp_path / name
    p.write_text(f"---\n{front}\n---\n\n{body}\n")
    return p


def test_parse_article_reads_front_matter(tmp_path):
    p = write_md(tmp_path, "a.md", "title: T\ncategory: basics\nsource: S\nreviewed: true")
    a = parse_article(p)
    assert (a.title, a.category, a.source, a.reviewed, a.slug) == ("T", "basics", "S", True, "a")


def test_parse_article_validates(tmp_path):
    (tmp_path / "x.md").write_text("no front matter")
    with pytest.raises(ValueError, match="front-matter"):
        parse_article(tmp_path / "x.md")
    p = write_md(tmp_path, "y.md", "title: T")
    with pytest.raises(ValueError, match="category"):
        parse_article(p)


def test_load_articles_recurses(tmp_path):
    (tmp_path / "c1").mkdir()
    write_md(tmp_path / "c1", "a.md", "title: A\ncategory: c1")
    write_md(tmp_path, "b.md", "title: B\ncategory: c2")
    assert {a.title for a in load_articles(tmp_path)} == {"A", "B"}


def test_chunker_splits_sections_and_windows():
    long = make_article("l", "Long", "basics", "## One\n" + "word " * 400 + "\n## Two\nshort")
    chunks = chunk_article(long, size=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 for c in chunks)
    assert chunks[-1].text.startswith("## Two")
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_article(ARTICLES[0], size=100, overlap=100)


def test_search_ranks_relevant_article_first(store):
    hits = store.search("what is compound interest", k=3)
    assert hits[0].chunk.slug == "compound"
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_category_filter_and_min_score(store):
    hits = store.search("interest", k=5, category="funds")
    assert hits and all(h.chunk.category == "funds" for h in hits)
    assert store.search("quantum chromodynamics zebra", k=5, min_score=0.5) == []
    assert store.search("   ") == []


def test_categories_and_len(store):
    assert store.categories == ["basics", "funds", "retirement"]
    assert len(store) >= 3


def test_save_and_load_roundtrip(store, tmp_path):
    store.save(tmp_path)
    loaded = VectorStore.load(HashEmbedder(), tmp_path)
    assert len(loaded) == len(store)
    assert loaded.search("roth ira", k=1)[0].chunk.slug == "roth"


def test_load_missing_index_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        VectorStore.load(HashEmbedder(), tmp_path)


def test_build_empty_rejected():
    with pytest.raises(ValueError):
        VectorStore.build([], HashEmbedder())


def test_retriever_context_and_deduped_sources(store):
    result = Retriever(store, top_k=4, min_score=0.0).retrieve("etf expense ratios")
    assert result.found
    assert result.context_block().startswith("[1] (")
    slugs = [s.title for s in result.sources()]
    assert len(slugs) == len(set(slugs))
