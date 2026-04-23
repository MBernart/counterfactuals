import json
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Option:
    id: str
    text: Dict[str, str]


@dataclass
class Question:
    id: str
    type: str
    options: List[Option] = field(default_factory=list)
    images: List[str] = field(default_factory=list)


@dataclass
class SurveyConfig:
    demographics: List[Question]
    ranking_questions: List[Question]
    outro: List[Question]


@dataclass
class QuestionResult:
    question_id: str
    type: str
    # Tracks option_id -> count (Guarantees zero-counts are included)
    counts: Dict[str, int] = field(default_factory=dict)
    # Tracks image_id -> rank_position (1 to 4) -> count
    ranking_counts: Dict[str, Dict[int, int]] = field(default_factory=dict)
    # Stores raw text for open questions
    open_answers: List[str] = field(default_factory=list)


def load_config(config_path: str) -> SurveyConfig:
    """Loads the JSON config and maps it into typed dataclasses."""
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def _parse_questions(q_list: List[dict]) -> List[Question]:
        parsed = []
        for q in q_list:
            opts = [
                Option(id=o['id'], text=o['text'])
                for o in q.get('options', [])
            ]
            imgs = [img['id'] for img in q.get('images', [])]

            # Default type inference if 'type' is missing (for info/welcome pages)
            q_type = q.get('type', 'ranking' if imgs else 'info')
            parsed.append(
                Question(id=q['id'], type=q_type, options=opts, images=imgs))
        return parsed

    return SurveyConfig(
        demographics=_parse_questions(data.get('demographics', [])),
        ranking_questions=_parse_questions(data.get('ranking_questions', [])),
        outro=_parse_questions(data.get('outro', [])))


def parse_results(config: SurveyConfig,
                  results_path: str) -> Dict[str, QuestionResult]:
    """Parses the CSV and aggregates data, respecting zero-vote options."""
    df = pd.read_csv(results_path)
    results: Dict[str, QuestionResult] = {}

    # 1. Parse Demographics (radio, select)
    for q in config.demographics:
        if q.type == 'info': continue

        # Initialize ALL options to 0
        res = QuestionResult(question_id=q.id,
                             type=q.type,
                             counts={o.id: 0
                                     for o in q.options})

        # Create reverse map from displayed text (any language) -> option ID
        text_to_id = {val: o.id for o in q.options for val in o.text.values()}

        if q.id in df.columns:
            for val in df[q.id].dropna():
                opt_id = text_to_id.get(
                    val, val)  # Fallback to val if it's already an ID
                if opt_id in res.counts:
                    res.counts[opt_id] += 1
        results[q.id] = res

    # 2. Parse Outro (checkbox, open)
    for q in config.outro:
        if q.type == 'info': continue

        res = QuestionResult(question_id=q.id, type=q.type)
        if q.type in ['checkbox', 'radio', 'select']:
            res.counts = {o.id: 0 for o in q.options}  # Initialize to 0
            if q.id in df.columns:
                for val in df[q.id].dropna():
                    if q.type == 'checkbox':
                        # Checkboxes in CSV are separated by ';'
                        for opt_id in str(val).split(';'):
                            if opt_id in res.counts:
                                res.counts[opt_id] += 1
                    else:
                        if val in res.counts:
                            res.counts[val] += 1
        elif q.type == 'open':
            if q.id in df.columns:
                res.open_answers = df[q.id].dropna().tolist()
        results[q.id] = res

    # 3. Parse Ranking Questions
    for q in config.ranking_questions:
        if not q.images: continue

        # Init 4 positions for every image to 0
        res = QuestionResult(
            question_id=q.id,
            type='ranking',
            ranking_counts={img: {
                1: 0,
                2: 0,
                3: 0,
                4: 0
            }
                            for img in q.images})

        for img in q.images:
            col_name = f"{q.id}_{img}_pos"
            if col_name in df.columns:
                counts = df[col_name].value_counts()
                for rank, count in counts.items():
                    if pd.notna(rank):
                        res.ranking_counts[img][int(rank)] += int(count)
        results[q.id] = res

    return results


if __name__ == "__main__":
    survey_config = load_config("config.json")
    parsed_results = parse_results(survey_config, "results.csv")

    # Example Output Check:
    print("Education Counts:", parsed_results['education'].counts)
    print("Q1 Ranking Breakdown:", parsed_results['q1'].ranking_counts)
