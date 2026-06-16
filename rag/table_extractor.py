import fitz
from langchain_core.documents import Document


SECTION_MAP = {
    "team profiles": ("team", "team_profiles"),
    "player batting statistics": ("batting", "batting_stats"),
    "player bowling statistics": ("bowling", "bowling_stats"),
    "head-to-head": ("h2h", "head_to_head"),
    "venue": ("venue", "venue"),
    "season": ("season", "season_results"),
    "recent form": ("form", "recent_form"),
    "ipl records": ("records", "records"),
}


def detect_section(page_text: str):
    text = page_text.lower()

    for key, value in SECTION_MAP.items():
        if key in text:
            return value

    return None, None


def build_metadata(section: str, row: dict):
    """
    Create metadata according to the dataset specification.
    """

    metadata = {
        "section": section
    }

    if section == "team":
        metadata["team_name"] = row.get("Team", "")
        metadata["season"] = 2024

    elif section == "batting":
        metadata["player_name"] = row.get("Player", "")
        metadata["team"] = row.get("Team", "")
        metadata["role"] = row.get("Role", "")

    elif section == "bowling":
        metadata["player_name"] = row.get("Player", "")
        metadata["team"] = row.get("Team", "")
        metadata["bowl_type"] = row.get("Type", "")

    elif section == "h2h":
        metadata["team1"] = row.get("Team 1", "")
        metadata["team2"] = row.get("Team 2", "")

    elif section == "venue":
        metadata["venue_name"] = row.get("Venue", "")
        metadata["city"] = row.get("City", "")
        metadata["pitch_type"] = row.get("Pitch Type", "")

    elif section == "season":
        metadata["team"] = row.get("Team", "")
        metadata["year"] = row.get("Year", "")

    elif section == "form":
        metadata["player_name"] = row.get("Player", "")
        metadata["season"] = 2024

    elif section == "records":
        metadata["category"] = row.get("Category", "")

    return metadata


def extract_tables(pdf_path: str):

    pdf = fitz.open(pdf_path)

    documents = []

    for page_number in range(len(pdf)):

        page = pdf[page_number]

        page_text = page.get_text("text")

        section, table_name = detect_section(page_text)

        if section is None:
            continue

        tables = page.find_tables()

        for table in tables:

            df = table.to_pandas()

            for _, row in df.iterrows():

                row_dict = row.to_dict()

                page_content = "\n".join(
                    f"{key}: {value}"
                    for key, value in row_dict.items()
                )

                metadata = build_metadata(section, row_dict)

                metadata["table_name"] = table_name
                metadata["page_number"] = page_number + 1

                documents.append(
                    Document(
                        page_content=page_content,
                        metadata=metadata
                    )
                )

    pdf.close()

    return documents