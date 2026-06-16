import fitz
from langchain_core.documents import Document
import pandas as pd

def detect_table_type(df):
    columns = [
        str(col).replace("\n", " ").strip().lower()
        for col in df.columns
    ]

    # Section 1 - Team Profiles (main table)
    if (
        "team" in columns
        and "captain" in columns
        and "coach" in columns
        and "home venue" in columns
    ):
        return "team", "team_profiles"

    # Section 1 - Team strategy table
    if (
        "team" in columns
        and "playing style & strategy" in columns
    ):
        return "team", "team_profiles"

    # Section 2 - Batting
    if "runs" in columns and "sr" in columns:
        return "batting", "batting_stats"

    # Section 3 - Bowling
    if "wkts" in columns and "econ" in columns:
        return "bowling", "bowling_stats"

    # Section 4 - Head-to-Head
    if (
        "team 1 wins" in columns
        and "team 2 wins" in columns
    ):
        return "h2h", "head_to_head"

    # Section 5 - Venue statistics
    if (
        "venue" in columns
        and "city" in columns
        and "pitch type" in columns
    ):
        return "venue", "venue_reports"

    # Section 5 - Venue narratives
    if (
        "venue narrative reports" in columns
        and "col1" in columns
    ):
        return "venue", "venue_reports"

    # Section 6 - Season performance
    if (
        "2019 pos" in columns
        and "2020 pos" in columns
    ):
        return "season", "season_results"

    # Section 7 - Recent form
    if "form trend" in columns:
        return "form", "recent_form"

    # Section 8 - IPL records
    if (
        "category" in columns
        and "record" in columns
        and "opponent" in columns
    ):
        return "records", "ipl_records"

    # Section 11 - Conflict table
    if (
        "primary source value" in columns
        and "secondary source value" in columns
    ):
        return "validation", "conflict_table"

    return None, None


def build_metadata(section, row):
    """
    Build metadata for a single row.
    Uses row.get(...) so missing columns won't raise errors.
    """
    row = {
    str(k).replace("\n", " ").strip(): v
    for k, v in row.items()
    }

    metadata = {
        "section": section,
    }

    if section in ("batting", "bowling", "form"):
        metadata["player_name"] = row.get("Player", "")
        metadata["team"] = row.get("Team", "")

    if section == "batting":
        metadata["role"] = row.get("Role", "")

    if section == "bowling":
        metadata["bowl_type"] = row.get("Type", "")

    if section == "team":
        metadata["team_name"] = row.get("Team", "")
        metadata["season"] = 2024

    if section == "venue":
        metadata["venue_name"] = (
            row.get("Venue")
            or row.get("Venue Narrative Reports")
            or ""
        )
        metadata["city"] = row.get("City", "")
        metadata["pitch_type"] = row.get("Pitch Type", "")

    if section == "h2h":
        metadata["team1"] = row.get("Team 1", "")
        metadata["team2"] = row.get("Team 2", "")

    if section == "records":
        metadata["category"] = row.get("Category", "")

    return metadata


def extract_tables(pdf_path: str):
    pdf = fitz.open(pdf_path)

    documents = []

    for page_number in range(len(pdf)):
        page = pdf[page_number]

        tables = page.find_tables()

        if len(tables.tables) == 0:
            continue

        print(f"\n========== Page {page_number + 1} ==========")

        for table_index, table in enumerate(tables.tables):
            df = table.to_pandas()
            if (
                page_number + 1 == 7
                and len(df.columns) == 2
            ):
                repaired_rows = [list(df.columns)] + df.values.tolist()

                df = pd.DataFrame(
                        repaired_rows,
                        columns=[
                        "Venue Narrative Reports",
                        "Col1"
                        ]
                        )

            print(f"\nTable {table_index + 1}")
            print(df.head())
            print("Columns:", df.columns.tolist())

            section, table_name = detect_table_type(df)

            if section is None:
                print("⚠️ Skipping unknown table")
                continue

            for _, row in df.iterrows():
                row_dict = row.to_dict()

                page_content = "\n".join(
                    f"{key}: {value}"
                    for key, value in row_dict.items()
                )

                metadata = build_metadata(section, row_dict)

                metadata["table_name"] = table_name
                metadata["page_number"] = page_number + 1
                metadata["table_index"] = table_index

                documents.append(
                    Document(
                        page_content=page_content,
                        metadata=metadata,
                    )
                )

    pdf.close()

    return documents