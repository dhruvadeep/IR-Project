import re


def calculate_flesch_reading_ease(text: str) -> float:
    """
    Calculate Flesch Reading Ease score.
    Formula: 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    """
    if not text:
        return 0.0

    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    num_sentences = len(sentences)

    words = re.findall(r"\b\w+\b", text)
    num_words = len(words)

    if num_words == 0 or num_sentences == 0:
        return 0.0

    # Simple syllable estimation
    def count_syllables(word):
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count

    num_syllables = sum(count_syllables(w) for w in words)

    score = (
        206.835
        - 1.015 * (num_words / num_sentences)
        - 84.6 * (num_syllables / num_words)
    )
    return round(score, 2)


def analyze_seo(title: str, body: str) -> dict:
    """
    Analyze SEO factors for a news article.
    """
    analysis = {}

    # 1. Title Analysis
    title_chars = len(title)

    analysis["title_length_check"] = {
        "value": title_chars,
        "status": "good" if 40 <= title_chars <= 60 else "warning",
        "message": "Optimal title length is 40-60 characters.",
    }

    # 2. Content Analysis
    word_count = len(body.split())
    analysis["word_count"] = {
        "value": word_count,
        "status": "good" if word_count >= 300 else "warning",
        "message": "Articles should ideally be > 300 words.",
    }

    # 3. Keyword Density (Title words in Body)
    # Simple check: how many title keywords appear in the first 100 words?
    title_keywords = [w.lower() for w in title.split() if len(w) > 3]
    body_start = body[:500].lower()  # First ~100 words

    found_keywords = [w for w in title_keywords if w in body_start]
    density_score = len(found_keywords) / len(title_keywords) if title_keywords else 0

    analysis["keyword_prominence"] = {
        "value": f"{len(found_keywords)}/{len(title_keywords)}",
        "status": "good" if density_score > 0.5 else "warning",
        "message": "Important keywords from title should appear in the first paragraph.",
    }

    # 4. Readability
    readability = calculate_flesch_reading_ease(body)
    readability_status = "good"
    if readability < 30:
        readability_status = "difficult"
    elif readability > 70:
        readability_status = "easy"

    analysis["readability"] = {
        "score": readability,
        "status": readability_status,
        "message": "Score 60-70 is standard. Lower is harder, higher is easier.",
    }

    return analysis
