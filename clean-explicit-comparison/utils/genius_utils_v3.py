import os
import time
import regex as re
import logging
import json
import string
import unicodedata
from dotenv import load_dotenv
from lyricsgenius import Genius
from requests.exceptions import HTTPError

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# These keywords are used to search Genius for clean versions of songs
CLEAN_KEYWORDS = ["clean", "radio edit", "censored"]

# Configure the API client
def setup_genius_client():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY is not set in the environment")
    return Genius(
        api_key,
        skip_non_songs=True,
        remove_section_headers=True,
        verbose=False,
        timeout=15,
        retries=3
    )


def save_lyrics_pair(base_title, version_label, raw_lyrics, cleaned_lyrics, out_dir="lyrics"):
    """
    Saves both raw and cleaned lyrics to separate text files.
    """
    os.makedirs(out_dir, exist_ok=True)
    safe_title = base_title.replace("/", "_").replace("\\", "_")

    raw_path = os.path.join(out_dir+"/raw", f"{safe_title}__{version_label}_raw.txt")
    clean_path = os.path.join(out_dir+"/cleaned", f"{safe_title}__{version_label}_cleaned.txt")

    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_lyrics)

    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(cleaned_lyrics)

    return raw_path, clean_path


def clean_lyrics(raw: str, debug=False) -> str:
    lines = raw.splitlines()
    clean = []
    found_lyrics_start = False

    for line in lines:
        line = line.strip()
        if not found_lyrics_start:
            if re.search(r"lyrics", line.lower()) or re.search(r"intro", line.lower()):
                found_lyrics_start = True
                continue
            if re.match(r"\d+\s+contributors", line.lower()):
                continue
            if re.search(r"translations", line.lower()):
                continue
            if re.search(r"read more", line.lower()):
                continue
            if re.match(r"[^\w]*$", line):
                continue
        else:
            if re.search(r"(русский|简体中文|português|srpski|italiano|deutsch|français|español)", line.lower()):
                continue
            clean.append(line)

    cleaned_text = "\n".join(clean).strip()
    if debug:
        print("Raw Lyrics (truncated):", raw[:300], "...")
        print("Cleaned Lyrics (truncated):", cleaned_text[:300], "...")
    return cleaned_text


LOOKALIKE_CHAR_MAP = {
    # Latin lookalikes
    "а": "a",  # Cyrillic a
    "ɑ": "a",  # Latin alpha
    "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "å": "a", "ā": "a", "ă": "a", "ą": "a",
    "А": "A",  # Cyrillic A
    "ß": "ss",
    "Ь": "",   # Cyrillic soft sign

    "е": "e",  # Cyrillic e
    "è": "e", "é": "e", "ê": "e", "ë": "e", "ē": "e", "ė": "e", "ę": "e",
    "Ε": "E",  # Greek Epsilon
    "е": "e",  # Cyrillic e

    "і": "i",  # Cyrillic i
    "í": "i", "ì": "i", "î": "i", "ï": "i", "ī": "i", "į": "i", "İ": "i", "ι": "i",
    "Ι": "I",  # Greek Capital Iota

    "о": "o",  # Cyrillic o
    "ô": "o", "ö": "o", "ò": "o", "ó": "o", "õ": "o", "ø": "o", "ō": "o",
    "Ο": "O", "ο": "o",  # Greek Omicron
    "Օ": "O",  # Armenian

    "р": "p",  # Cyrillic p
    "Р": "P",

    "ѕ": "s",  # Cyrillic s
    "ș": "s", "š": "s", "ś": "s", "ß": "ss",

    "т": "t",  # Cyrillic t
    "ţ": "t", "ť": "t", "ŧ": "t",

    "υ": "u", "ù": "u", "ú": "u", "û": "u", "ü": "u", "ū": "u", "ų": "u", "ű": "u",
    "μ": "u",  # visually similar
    "ս": "u",  # Armenian small letter se

    "х": "x",  # Cyrillic x
    "Χ": "X",  # Greek chi

    "у": "y",  # Cyrillic y
    "ý": "y", "ÿ": "y", "ŷ": "y",

    "з": "3",  # Cyrillic ze
    "₃": "3", "③": "3",

    "ς": "s", "σ": "s",  # Greek sigmas
    "ϲ": "c",  # Greek lunate sigma

    "ç": "c", "ć": "c", "č": "c", "ĉ": "c",

    "ñ": "n", "ń": "n", "ň": "n", "ŋ": "n",

    "ℓ": "l", "ł": "l",

    "℮": "e",

    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "`": "'", "´": "'", "′": "'",

    "–": "-", "—": "-", "−": "-",  # various dashes
    "…": "...",

    "ⅰ": "i", "Ⅰ": "I",
    "ⅱ": "ii", "Ⅱ": "II",
    "ⅲ": "iii", "Ⅲ": "III",
    "ⅳ": "iv", "Ⅳ": "IV",
    "ⅴ": "v", "Ⅴ": "V",

    # Symbols often mistaken
    "∅": "0", "〇": "0", "Ｏ": "O",
    "①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5", "⑥": "6", "⑦": "7", "⑧": "8", "⑨": "9",
}

def normalize_lookalike_chars(text: str) -> str:
    """
    Replace non-ASCII Unicode lookalike characters with their ASCII equivalents.
    """
    normalized = []
    for char in text:
        if char in LOOKALIKE_CHAR_MAP:
            normalized.append(LOOKALIKE_CHAR_MAP[char])
        else:
            normalized.append(char)
    return ''.join(normalized)


import regex as re

def remove_punctuation_and_blank_lines(text: str) -> str:
    """
    Normalizes homoglyphs, lowercases text, removes all punctuation (ASCII + Unicode),
    preserves blank lines, removes common stopwords, and filters filler words.
    """
    STOPWORDS = {
        "the", "and", "a", "an", "to", "in", "of", "for", "on", "at", "with", "by", "tha", "ooh", "woo", "oh", "ah", "agh"
    }

    PRONOUNS = {
        # Standard pronouns
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",

        # Slang / colloquial forms
        "yall", "ya", "yo", "em",

        # Contractions (apostrophes removed)
        "im", "ive", "ill", "id",
        "youre", "youve", "youll", "youd",
        "hes", "hell", "hed",
        "shes", "shell", "shed",
        "were", "weve", "well", "wed",
        "theyre", "theyve", "theyll", "theyd",
        "itll", "itd"

        # Sound effects/onomatopoeia
        'pfft', 'pft', 'bzz', 'ugh', 'kshh', 'eugh', 'tsk', 'tsktsk', 'pssh', 
        'oof', 'uh', 'uhh', 'uhuh', 'hmm', 'hm', 'mmm', 'grr', 'agh', 'ahh', 
        'bleh', 'blah', 'sniff', 'gasp', 'cough', 'burp', 'hiccup', 'sigh', 'shh'
    
    }

    def is_filler_word(word):
        word = re.sub(r"\p{P}+", "", word.lower())
        if len(word) == 2 and word[-1] in {"a", "o"}:
            return True
        if re.fullmatch(r"([a-z]{2,4})\1{1,}", word):  # haha, hahaha, etc.
            return True
        if re.fullmatch(r"(na|la|ha|da|ba){2,}", word):
            return True
        return False

    # Normalize homoglyphs and lowercase
    text = normalize_lookalike_chars(text).lower()

    # Remove all Unicode punctuation
    text = re.sub(r"\p{P}+", "", text)

    # Process each line independently
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        words = line.strip().split()
        filtered_words = [
        w for w in words
        if w not in STOPWORDS and w not in PRONOUNS and not is_filler_word(w)
        ]
        if filtered_words:
            cleaned_lines.append(" ".join(filtered_words))  # Add only non-empty cleaned lines

    return "\n".join(cleaned_lines)




def safe_search_song(genius, query, artist=None, delay=1.2, retries=4):
    for i in range(retries):
        try:
            result = genius.search_song(query, artist, get_full_info=False)
            time.sleep(delay)
            return result
        except HTTPError as e:
            if "429" in str(e):
                wait_time = 2 ** i
                logger.warning(f"Rate limit hit. Sleeping for {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"HTTP error for {query}: {e}")
                return None
        except Exception as e:
            logger.error(f"General error for {query}: {e}")
            return None
    return None


def find_versions(genius, song_title, artist, debug=False):
    from .genius_utils_v3 import CLEAN_KEYWORDS  # avoid circular import issues

    base = safe_search_song(genius, song_title, artist)
    candidates = [base] if base else []

    for keyword in CLEAN_KEYWORDS:
        alt_title = f"{song_title} ({keyword})"
        alt = safe_search_song(genius, alt_title, artist)
        if alt:
            candidates.append(alt)

    versions = {}
    for s in candidates:
        if s is None:
            continue
        label = "clean" if any(k in s.title.lower() for k in CLEAN_KEYWORDS) else "explicit"
        if label not in versions:
            try:
                raw_lyrics = s.lyrics
                cleaned = clean_lyrics(raw_lyrics, debug=debug)
                versions[label] = (s.title, cleaned, s.url)
            except Exception as e:
                logger.warning(f"Couldn't fetch lyrics for {s.title}: {e}")
        if len(versions) == 2:
            break

    return versions if len(versions) > 1 else None


def get_artist_songs_metadata(genius, artist_name, max_songs=20):
    try:
        return genius.search_artist(artist_name, max_songs=max_songs, get_full_info=False)
    except Exception as e:
        logger.error(f"Failed to fetch artist metadata: {e}")
        return None


def get_clean_explicit_pairs(genius, artist_name, max_songs=20, debug=False):
    artist_obj = get_artist_songs_metadata(genius, artist_name, max_songs=max_songs)
    if not artist_obj:
        return []

    matched = []
    seen_titles = set()

    for song in artist_obj.songs:
        title = song.title.strip()
        if title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())

        logger.info(f"Scanning: {title}")
        versions = find_versions(genius, title, artist_name, debug=debug)
        # Sleep between full song iterations (for safety)
        time.sleep(1.5)

        if versions:
            matched.append({
                "base_title": title,
                "explicit": versions.get("explicit"),
                "clean": versions.get("clean")
            })

    return matched


def clean_lyrics_text(lyrics):
    # Normalize ellipsis
    lyrics = lyrics.replace("…", "...")

    # Remove any line that ends in "... Read More" or "Read More"
    lyrics = re.sub(
    r"^.*(?:\.{3}|…)\s*Read More\s*$", 
    "", 
    lyrics, 
    flags=re.IGNORECASE | re.MULTILINE
)

    # (Optional) remove standalone section headers
    section_pattern = r"^\s*(verse|chorus|intro|bridge|refrain|hook|outro)(\s+\d+)?(\s*\([^)]+\))?\s*:\s*$"
    lyrics = re.sub(section_pattern, "", lyrics, flags=re.IGNORECASE | re.MULTILINE)

    # Collapse multiple blank lines
    lyrics = re.sub(r"\n{2,}", "\n", lyrics)

    # Remove repeated words (e.g., "back back", "yeah yeah")
    def remove_repeats(match):
        return match.group(1)

    # Match two identical words separated by whitespace or punctuation
    lyrics = re.sub(r'\b(\w+)(?:[\s.,;:!?-]+)\1\b', remove_repeats, lyrics, flags=re.IGNORECASE)
    
    return lyrics.strip()


def save_raw_and_cleaned_lyrics(base_title, version_label, raw_lyrics, cleaned_lyrics, out_dir="lyrics"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir+"/raw", exist_ok=True)
    os.makedirs(out_dir+"/clean", exist_ok=True)
    safe_title = base_title.replace("/", "_").replace("\\", "_")

    raw_path = os.path.join(out_dir+"/raw", f"{safe_title}__{version_label}_raw.txt")
    clean_path = os.path.join(out_dir+"/clean", f"{safe_title}__{version_label}_cleaned.txt")

    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_lyrics)

    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(cleaned_lyrics)

    return raw_path, clean_path


def save_pairs_to_json(pairs, filename="eminem_clean_explicit_pairs.json", out_dir="lyrics"):
    data = []

    for pair in pairs:
        # Only save if both clean and explicit versions exist
        if not pair.get("explicit") or not pair.get("clean"):
            continue

        title_base = pair["base_title"]

        # Get raw lyrics
        raw_explicit = pair["explicit"][1]
        raw_clean = pair["clean"][1]

        # Clean them
        cleaned_explicit = clean_lyrics_text(raw_explicit)
        cleaned_clean = clean_lyrics_text(raw_clean)

        # Save raw and cleaned .txt files
        save_raw_and_cleaned_lyrics(title_base, "explicit", raw_explicit, cleaned_explicit, out_dir)
        save_raw_and_cleaned_lyrics(title_base, "clean", raw_clean, cleaned_clean, out_dir)

        # Add to JSON structure
        data.append({
            "title_base": title_base,
            "explicit": {
                "title": pair["explicit"][0],
                "lyrics": cleaned_explicit,
                "url": pair["explicit"][2],
            },
            "clean": {
                "title": pair["clean"][0],
                "lyrics": cleaned_clean,
                "url": pair["clean"][2],
            }
        })
    if not data:
        print("⚠️ No clean-explicit song pairs found. Nothing saved.")
        return
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n Exported {len(data)} clean-explicit song pairs to {filename}")
    print(f"Raw and cleaned .txt files saved to '{out_dir}/'")