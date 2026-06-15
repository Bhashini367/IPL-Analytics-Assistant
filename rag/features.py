import re

def extract_number(text, keyword):
    pattern = rf"{keyword}[^0-9]*([0-9]+\.?[0-9]*)"
    match = re.search(pattern, text.lower())

    if match:
        return float(match.group(1))

    return 0

def extract_batting_features(text):
    return {
        "runs": extract_number(text, "runs"),
        "strike_rate": extract_number(text, "strike"),
        "average": extract_number(text, "average")
    }