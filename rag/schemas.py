from dataclasses import dataclass


@dataclass
class ExtractedChunk:

    text: str

    metadata: dict