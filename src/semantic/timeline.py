from src.semantic.similarity import similar_articles
from src.indexer import load_indices
from datetime import datetime


def parse_date(date_str):
    """Safely parse date from the CCNews dataset."""
    try:
        return datetime.fromisoformat(date_str.replace("Z", ""))
    except:
        return None


def build_story_timeline(doc_id: int, top_k: int = 8):
    """
    Builds a chronological story timeline using similar articles.
    """
    # 1. Fetch article itself + similar articles
    _, _, doc_map = load_indices()

    main_doc = doc_map.get(doc_id)
    if not main_doc:
        return {"error": "Invalid doc_id"}

    main_title, main_body, main_site, main_date = main_doc[1], main_doc[2], main_doc[3], main_doc[4]

    similar = similar_articles(doc_id, top_k)

    # 2. Parse dates
    items = []
    for art in similar:
        d = doc_map[art["doc_id"]]
        title, body, site, date = d[1], d[2], d[3], d[4]
        parsed = parse_date(date)

        items.append({
            "doc_id": art["doc_id"],
            "title": title,
            "snippet": body[:150] + "...",
            "site": site,
            "date": date,
            "parsed_date": parsed,
        })

    # 3. Sort by date
    items = [i for i in items if i["parsed_date"] is not None]
    items.sort(key=lambda x: x["parsed_date"])

    if not items:
        return {"error": "No valid dates found for timeline"}

    # 4. Assign timeline labels
    n = len(items)
    output = []
    for idx, item in enumerate(items):
        if idx == 0:
            label = "Breaking"
        elif idx < n * 0.6:
            label = "Update"
        elif len(item["snippet"]) > 120:
            label = "Analysis"
        else:
            label = "Aftermath"

        item["label"] = label
        output.append(item)

    # 5. Return main + timeline
    return {
        "main_article": {
            "doc_id": doc_id,
            "title": main_title,
            "date": main_date,
            "site": main_site,
            "snippet": main_body[:150] + "..."
        },
        "timeline": output
    }
