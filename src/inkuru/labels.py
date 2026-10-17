"""The 14 KinNews sections, in the order of classes.txt (index 1..14)."""

LABELS = [
    "politics",
    "sport",
    "economy",
    "health",
    "entertainment",
    "history",
    "technology",
    "tourism",
    "culture",
    "fashion",
    "religion",
    "environment",
    "education",
    "relationship",
]

# Filled from data/kin_labels.json, which notebook 01 writes from the most common kin_label in the raw files.
# English names are the dataset's own en_label, not a fresh translation.
KIN = {label: label for label in LABELS}

BLURB = {
    "politics": "Government, elections, and public affairs.",
    "sport": "Matches, clubs, and competitions.",
    "economy": "Business, markets, and livelihoods.",
    "health": "Hospitals, disease, and public health.",
    "entertainment": "Music, film, and celebrity.",
    "history": "The past, memory, and commemoration.",
    "technology": "Devices, telecoms, and digital services.",
    "tourism": "Travel, places, and hospitality.",
    "culture": "Arts, language, and custom.",
    "fashion": "Clothes, style, and beauty.",
    "religion": "Faith, churches, and religious life.",
    "environment": "Land, climate, and conservation.",
    "education": "Schools, exams, and learning.",
    "relationship": "Family, marriage, and personal life.",
}


def load_kin(path) -> None:
    import json
    if not path.exists():
        return
    found = json.loads(path.read_text())
    for label, name in found.items():
        if label in KIN and name:
            KIN[label] = name
