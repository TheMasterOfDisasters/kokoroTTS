"""
German text normalization for TTS.

Pure Python, zero external dependencies. DEG2P uses this before handing text to
espeak-ng so common German dates, amounts, abbreviations, and numbers are read
as words instead of raw punctuation-heavy text.
"""

from __future__ import annotations

import re

_ONES = [
    "",
    "ein",
    "zwei",
    "drei",
    "vier",
    "fünf",
    "sechs",
    "sieben",
    "acht",
    "neun",
    "zehn",
    "elf",
    "zwölf",
    "dreizehn",
    "vierzehn",
    "fünfzehn",
    "sechzehn",
    "siebzehn",
    "achtzehn",
    "neunzehn",
]
_TENS = [
    "",
    "",
    "zwanzig",
    "dreißig",
    "vierzig",
    "fünfzig",
    "sechzig",
    "siebzig",
    "achtzig",
    "neunzig",
]


def int_to_de(n: int) -> str:
    """Convert an integer to German words."""
    if n < 0:
        return "minus " + int_to_de(-n)
    if n == 0:
        return "null"
    if n < 20:
        return _ONES[n]
    if n < 100:
        ones, tens = n % 10, n // 10
        return (_ONES[ones] + "und" + _TENS[tens]) if ones else _TENS[tens]
    if n < 1_000:
        hundreds, rest = divmod(n, 100)
        return ("ein" if hundreds == 1 else _ONES[hundreds]) + "hundert" + (int_to_de(rest) if rest else "")
    if n < 1_000_000:
        thousands, rest = divmod(n, 1_000)
        return ("ein" if thousands == 1 else int_to_de(thousands)) + "tausend" + (int_to_de(rest) if rest else "")
    if n < 1_000_000_000:
        millions, rest = divmod(n, 1_000_000)
        head = "eine Million" if millions == 1 else int_to_de(millions) + " Millionen"
        return head + (" " + int_to_de(rest) if rest else "")
    billions, rest = divmod(n, 1_000_000_000)
    head = "eine Milliarde" if billions == 1 else int_to_de(billions) + " Milliarden"
    return head + (" " + int_to_de(rest) if rest else "")


_ORD_IRREGULAR = {1: "erst", 2: "zweit", 3: "dritt", 7: "siebt", 8: "acht"}


def ordinal_stem_de(n: int) -> str:
    """Return the uninflected German ordinal stem."""
    if n in _ORD_IRREGULAR:
        return _ORD_IRREGULAR[n]
    return int_to_de(n) + ("t" if n < 20 else "st")


def year_de(n: int) -> str:
    if 1100 <= n <= 1999:
        century, rest = divmod(n, 100)
        return int_to_de(century) + "hundert" + (int_to_de(rest) if rest else "")
    return int_to_de(n)


_ABBREVS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bDr\.(?=\s)"), "Doktor"),
    (re.compile(r"\bProf\.(?=\s)"), "Professor"),
    (re.compile(r"\bHrn?\.\s"), "Herrn "),
    (re.compile(r"\bFr\.(?=\s[A-ZÄÖÜ])"), "Frau"),
    (re.compile(r"\bDipl\.\s*-?\s*Ing\."), "Diplom-Ingenieur"),
    (re.compile(r"\bMag\.(?=\s)"), "Magister"),
    (re.compile(r"[Ss]tr\.(?=\s)"), "Straße"),
    (re.compile(r"\bNr\.(?=\s*\d)"), "Nummer"),
    (re.compile(r"\bTel\.(?=\s)"), "Telefon"),
    (re.compile(r"\bAbt\.(?=\s)"), "Abteilung"),
    (re.compile(r"\bGmbH\b"), "Gesellschaft mit beschränkter Haftung"),
    (re.compile(r"\bAG\b(?=[\s,.]|$)"), "Aktiengesellschaft"),
    (re.compile(r"\bz\.\s*B\."), "zum Beispiel"),
    (re.compile(r"\bd\.\s*h\."), "das heißt"),
    (re.compile(r"\bu\.\s*a\."), "unter anderem"),
    (re.compile(r"\bbzw\."), "beziehungsweise"),
    (re.compile(r"\busw\."), "und so weiter"),
    (re.compile(r"\betc\.", re.I), "et cetera"),
    (re.compile(r"\bca\."), "circa"),
    (re.compile(r"\bvgl\."), "vergleiche"),
    (re.compile(r"\binkl\."), "inklusive"),
    (re.compile(r"\bexkl\."), "exklusive"),
    (re.compile(r"\bggf\."), "gegebenenfalls"),
    (re.compile(r"\bi\.\s*d\.\s*R\."), "in der Regel"),
    (re.compile(r"\bo\.\s*ä\."), "oder ähnliches"),
    (re.compile(r"\bu\.\s*U\."), "unter Umständen"),
    (re.compile(r"\bJan\.(?=\s)"), "Januar"),
    (re.compile(r"\bFeb\.(?=\s)"), "Februar"),
    (re.compile(r"\bMär\.(?=\s)"), "März"),
    (re.compile(r"\bApr\.(?=\s)"), "April"),
    (re.compile(r"\bJun\.(?=\s)"), "Juni"),
    (re.compile(r"\bJul\.(?=\s)"), "Juli"),
    (re.compile(r"\bAug\.(?=\s)"), "August"),
    (re.compile(r"\bSep\.(?=\s)"), "September"),
    (re.compile(r"\bOkt\.(?=\s)"), "Oktober"),
    (re.compile(r"\bNov\.(?=\s)"), "November"),
    (re.compile(r"\bDez\.(?=\s)"), "Dezember"),
]

_DE_MONTHS = [
    "",
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
]

_CURRENCY_SYM = {"€": "Euro", "$": "Dollar", "£": "Pfund", "¥": "Yen"}
_CURRENCY_PAT = re.compile(r"([€$£¥])\s*(\d[\d.,]*)|(\d[\d.,]*)\s*([€$£¥])")
_GER_NUM = re.compile(r"\b\d{1,3}(?:\.\d{3})*(?:,\d+)?\b|\b\d+,\d+\b|\b\d+\b")
_YEAR_RE = re.compile(r"\b(\d{4})\b")
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_ORD_RE = re.compile(r"(?<!\n)(?<!\d)(\d{1,4})\.\s")


def _currency_repl(match: re.Match[str]) -> str:
    symbol, number = (match.group(1), match.group(2)) if match.group(1) else (match.group(4), match.group(3))
    word = _CURRENCY_SYM.get(symbol, symbol)
    cleaned = number.replace(".", "").replace(",", ".")
    try:
        value = float(cleaned)
    except ValueError:
        return match.group(0)
    units = int(value)
    cents = round((value - units) * 100)
    if cents == 0:
        return int_to_de(units) + " " + word
    return int_to_de(units) + " " + word + " und " + int_to_de(cents) + " Cent"


def _time_repl(match: re.Match[str]) -> str:
    hours, minutes = int(match.group(1)), int(match.group(2))
    return int_to_de(hours) + " Uhr" + (" " + int_to_de(minutes) if minutes else "")


def _date_repl(match: re.Match[str]) -> str:
    day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return match.group(0)
    return ordinal_stem_de(day) + "e " + _DE_MONTHS[month] + " " + year_de(year)


def _ordinal_repl(match: re.Match[str]) -> str:
    return ordinal_stem_de(int(match.group(1))) + "e "


def _number_repl(match: re.Match[str]) -> str:
    raw = match.group(0)
    cleaned = raw.replace(".", "").replace(",", ".")
    try:
        if "." in cleaned:
            int_part, frac = cleaned.split(".", 1)
            digits = " ".join(int_to_de(int(digit)) for digit in frac)
            return int_to_de(int(int_part)) + " Komma " + digits
        return int_to_de(int(cleaned))
    except (ValueError, OverflowError):
        return raw


def normalize_text_de(text: str) -> str:
    """Normalize common German text constructs for TTS phonemization."""
    if not text:
        return text

    text = (
        text.replace("\u201e", '"')
        .replace("\u201c", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u00ab", '"')
        .replace("\u00bb", '"')
        .replace("\u2039", '"')
        .replace("\u203a", '"')
    )
    text = re.sub(r"[^\S \n]", " ", text)
    for pattern, replacement in _ABBREVS:
        text = pattern.sub(replacement, text)
    text = _CURRENCY_PAT.sub(_currency_repl, text)
    text = _TIME_RE.sub(_time_repl, text)
    text = _DATE_RE.sub(_date_repl, text)
    text = _ORD_RE.sub(_ordinal_repl, text)

    def _year_or_num(match: re.Match[str]) -> str:
        year = int(match.group(1))
        return year_de(year) if 1100 <= year <= 2099 else int_to_de(year)

    text = _YEAR_RE.sub(_year_or_num, text)
    text = _GER_NUM.sub(_number_repl, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
