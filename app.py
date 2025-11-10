# app.py — Flipdish Menu Builder (client-facing)
# Hidden intelligence:
# - PDF raster at 300 DPI
# - Cached page extraction
# - Silent learning loop: examples.jsonl (few-shot retrieval)
# - Optional rules.json auto-applied (no UI)
# - Conditional modifiers supported (groups can live under options)

import base64
import io
import json
import os
import uuid
import datetime
import re
import time
import math
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
from PIL import Image, ImageFile

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# --------- basic setup ---------
ImageFile.LOAD_TRUNCATED_IMAGES = True
load_dotenv()
st.set_page_config(page_title="Flipdish Menu Builder", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.sidebar.warning("Set OPENAI_API_KEY in environment or secrets to enable AI extraction.")

if OpenAI is not None and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# ============================= Utils ==============================

def guid() -> str:
    return str(uuid.uuid4())

def now_iso_z() -> str:
    # e.g. 2024-01-01T12:00:00Z
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def now_iso_hms() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat()

def to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

def smart_float(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = s.replace(",", "")
    m = re.match(r"^\$?\s*([0-9]+(?:\.[0-9]+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

PRICE_RE = re.compile(r"(\d+(?:\.\d{1,2})?)")

def parse_price_from_text(*texts: str) -> Optional[float]:
    for t in texts:
        if not t:
            continue
        for m in PRICE_RE.finditer(t):
            val = smart_float(m.group(1))
            if val is not None:
                return val
    return None

def clean_caption(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t

SMALL_WORDS = {
    "a","an","and","as","at","but","by","en","for","if","in","of","on","or",
    "the","to","vs","via","with"
}

def _cap_hyphenated(word: str) -> str:
    parts = word.split("-")
    return "-".join(p.capitalize() for p in parts)

def smart_title(text: str) -> str:
    if not text:
        return text
    if text.isupper():
        # Already shouting, but we fix casing softly
        text = text.lower()
    if " " not in text and "-" not in text:
        return text[:1].upper() + text[1:]
    if any(c.islower() for c in text[1:]) and any(c.isupper() for c in text[1:]):
        # mixed case, probably fine
        return text
    tokens = re.split(r'(\s+)', text.strip())
    words_only = [t for t in tokens if not re.match(r'\s+', t)]
    result = []
    word_index = 0
    for t in tokens:
        if re.match(r'\s+', t):
            result.append(t)
            continue
        lower = t.lower()
        if t.isupper() and len(t) > 1 and "-" not in t:
            out = t
        else:
            base = _cap_hyphenated(lower)
            if (word_index == 0 or word_index == len(words_only) - 1 or lower not in SMALL_WORDS):
                out = base[0].upper() + base[1:] if base else base
            else:
                out = base.lower()
        result.append(out)
        word_index += 1
    return "".join(result)

# Category colors (optional – looks nice if Flipdish shows them)
CATEGORY_COLOR_RULES = [
    (re.compile(r"^(starter|starters)$", re.I), "#E67E22"),
    (re.compile(r"^(main|mains)$", re.I), "#C0392B"),
    (re.compile(r"^(side|sides)$", re.I), "#FBC02D"),
    (re.compile(r"^(soup|soups)\s*/\s*(salad|salads)$", re.I), "#2E8B57"),
    (re.compile(r"^(soup|soups)$", re.I), "#2E8B57"),
    (re.compile(r"^(salad|salads)$", re.I), "#2E8B57"),
    (re.compile(r"^(dessert|desserts)$", re.I), "#8E44AD"),
    (re.compile(r"^(beverage|beverages|drink|drinks)$", re.I), "#9b9b9b"),
]

def infer_category_color(name: str) -> Optional[str]:
    n = (name or "").strip()
    for rx, col in CATEGORY_COLOR_RULES:
        if rx.match(n):
            return col
    return None

# Separator patterns for interpreting inline notes vs caption
_SPLIT_PATTERNS = [
    re.compile(r"^(?P<name>.+?)\s*[-–—]\s*(?P<desc>.+)$"),
    re.compile(r"^(?P<name>.+?)\s*\(\s*(?P<desc>.+)\)$"),
]

def split_caption_and_inline_notes(text: str) -> Tuple[str, str]:
    t = (text or "").strip()
    for p in _SPLIT_PATTERNS:
        m = p.match(t)
        if m:
            return m.group("name").strip(), m.group("desc").strip()
    return t, ""

# ============================== Allergen detection ==============================

# Canonical allergen tags aligned with Flipdish-style dietaryTags usage.
ALLERGEN_KEYWORDS = {
    "Celery": ["celery"],
    "Crustaceans": ["crustacean", "crustaceans", "prawn", "prawns", "shrimp", "shrimps", "crab", "lobster", "langoustine"],
    "Fish": ["fish", "anchovy", "anchovies", "salmon", "tuna", "cod", "haddock"],
    "Gluten": ["gluten"],
    "Wheat": ["wheat", "breadcrumbs", "breaded"],
    "Lupin": ["lupin", "lupine"],
    "Milk": ["milk", "butter", "cheese", "cream", "yoghurt", "yogurt", "mozzarella", "cheddar", "parmesan", "mascarpone"],
    "Molluscs": ["mollusc", "molluscs", "mussel", "mussels", "clam", "clams", "oyster", "oysters", "scallop", "scallops"],
    "Mustard": ["mustard"],
    "Nuts": [
        "nut ", " nuts", "walnut", "walnuts", "almond", "almonds",
        "hazelnut", "hazelnuts", "pistachio", "pistachios",
        "pecan", "pecans", "cashew", "cashews"
    ],
    "Peanuts": ["peanut", "peanuts"],
    "Sesame": ["sesame", "tahini"],
    "Soya": ["soya"],
    "Soybeans": ["soy", "soybean", "soybeans", "tofu", "edamame"],
    "Sulphur Dioxide": ["sulphur dioxide", "sulfur dioxide", "sulphites", "sulfites", "e220", "e221", "e222", "e223", "e224", "e226", "e227", "e228"],
    "Egg": ["egg", "eggs", "mayonnaise", "mayo", "aioli", "hollandaise"],
    "Alcohol": ["alcohol", "beer", "wine", "cider", "vodka", "rum", "gin", "whiskey", "whisky", "liqueur", "brandy"],
}

# Extra heuristic mappings for branded / obvious ingredients
BRAND_ALLERGEN_HINTS = {
    "nutella": ["Nuts", "Milk"],
    "biscoff": ["Gluten", "Wheat", "Soya", "Soybeans"],
}

def _detect_allergens_from_text(*parts: str) -> list:
    """
    Lightweight heuristic:
    - Look for known allergen words/synonyms in item / modifier text.
    - Returns a sorted list of unique canonical allergen names.
    """
    text = " ".join([p or "" for p in parts]).lower()
    text_spaced = f" {text} "

    found = set()

    # Brand / special hints
    for brand, tags in BRAND_ALLERGEN_HINTS.items():
        if brand in text:
            found.update(tags)

    # Keyword-based detection
    for label, keywords in ALLERGEN_KEYWORDS.items():
        for kw in keywords:
            # word-boundary-ish match; some kws contain spaces or trailing chars
            if re.search(r"\b" + re.escape(kw) + r"\b", text) or kw in text_spaced:
                found.add(label)
                break

    return sorted(found)

def _attach_allergens_to_params(existing_params_json: str, allergens: list) -> str:
    """
    Merge detected allergens into paramsJson as:
      dietaryConfiguration.dietaryTags = "A,B,C"
    preserving any existing paramsJson content.
    """
    if not allergens:
        return existing_params_json or ""

    try:
        params = json.loads(existing_params_json) if existing_params_json else {}
        if not isinstance(params, dict):
            params = {}
    except Exception:
        params = {}

    dietary = params.get("dietaryConfiguration", {})
    existing_tags = set(
        t.strip() for t in str(dietary.get("dietaryTags", "")).split(",") if t.strip()
    )
    existing_tags.update(allergens)
    dietary["dietaryTags"] = ",".join(sorted(existing_tags))
    params["dietaryConfiguration"] = dietary

    return json.dumps(params, ensure_ascii=False)

# ============================== Learning store ==============================

EXAMPLES_PATH = "examples.jsonl"  # newline-delimited examples
DEFAULT_RULES = {
    "modifier_caption_aliases": {
        "ADD": ["EXTRAS", "ADD-ONS", "GOES WELL WITH", "SIDES"],
        "CHOOSE PROTEIN": ["PROTEIN CHOICE", "CHOOSE YOUR PROTEIN"]
    },
    "force_minmax": {
        # "CHOOSE PROTEIN": [1, 1],
    },
    "option_aliases": {
        "bbq sauce": ["barbecue", "bbq"],
        "fries": ["french fries", "chips"]
    }
}

def save_example(source_snippet: str, flipdish_piece: dict, tags: List[str]):
    rec = {
        "ts": int(time.time()),
        "source": (source_snippet or "")[:4000],
        "flipdish": flipdish_piece,
        "tags": tags or []
    }
    try:
        with open(EXAMPLES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def load_examples() -> List[dict]:
    if not os.path.exists(EXAMPLES_PATH):
        return []
    out = []
    try:
        with open(EXAMPLES_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return []
    return out

def _tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())

def _bow(text):
    d = {}
    for tok in _tokenize(text):
        d[tok] = d.get(tok, 0) + 1
    return d

def _cos(a, b):
    num = sum(a.get(k, 0) * b.get(k, 0) for k in set(a) | set(b))
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0

def top_k_examples(query_text, k=3):
    q = _bow(query_text or "")
    if not q:
        return []
    ex = load_examples()
    scored = []
    for e in ex:
        fq = _bow(e.get("source") or "")
        c = _cos(q, fq)
        if c > 0:
            scored.append((c, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:k]]

def try_load_rules() -> dict:
    # Load rules.json if present, merge onto DEFAULT_RULES
    rules = DEFAULT_RULES.copy()
    try:
        if os.path.exists("rules.json"):
            with open("rules.json", encoding="utf-8") as f:
                user_rules = json.load(f)
            if isinstance(user_rules, dict):
                for k, v in user_rules.items():
                    if isinstance(v, dict) and isinstance(rules.get(k), dict):
                        nv = rules[k].copy()
                        nv.update(v)
                        rules[k] = nv
                    else:
                        rules[k] = v
    except Exception:
        pass
    return rules

# ============================== PDF / image helpers ==============================

def read_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not available on server.")
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        blocks = page.get_text("blocks")
        pages.append({
            "number": i + 1,
            "text": text,
            "blocks": blocks,
        })
    return pages

def raster_page_to_png(page, dpi=300) -> bytes:
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")

def find_item_rects(page, name: str):
    if not name:
        return []
    rects = []
    for inst in page.search_for(name, quads=False):
        rects.append(inst)
    return rects

def nearest_image_crop(page, item_rect, margin=4):
    images = page.get_images(full=True)
    if not images:
        return None
    best = None
    best_dist = None
    for img in images:
        xref = img[0]
        rect = fitz.Rect(img[5], img[6], img[7], img[8]) if len(img) >= 9 else None
        if not rect:
            continue
        cx = (rect.x0 + rect.x1) / 2
        cy = (rect.y0 + rect.y1) / 2
        dx = max(item_rect.x0 - cx, 0, cx - item_rect.x1)
        dy = max(item_rect.y0 - cy, 0, cy - item_rect.y1)
        dist = (dx * dx + dy * dy) ** 0.5
        if best is None or dist < best_dist:
            best = rect
            best_dist = dist
    if best is None:
        return None
    clip = fitz.Rect(best.x0 - margin, best.y0 - margin, best.x1 + margin, best.y1 + margin)
    pix = page.get_pixmap(clip=clip, dpi=300)
    return pix.tobytes("png")

# ============================== Regex fallback for missed modifiers ==============================

MODIFIER_HEADERS = [
    r"goes\s+well\s+with", r"goes\s+with", r"add[-\s]*ons?", r"addons?", r"extras?",
]
MODIFIER_HEADER_RE = re.compile(r"(?i)\b(" + "|".join(MODIFIER_HEADERS) + r")\b[:\s]*")
PLUS_PRICE_LINE = re.compile(r'(?i)^(?P<name>.*?\S)\s*(?:\+|plus\s*)(?P<price>\d+(?:\.\d+)?)\s*$')
ADD_PATTERN = re.compile(r"(?i)\badd\s+(?P<opts>[^.;\n/]+?)\s*(?:\+\s*(?P<price>\d+(?:\.\d+)?))(?=[\s\).,;/]|$)")
CHOICE_PATTERN = re.compile(r"(?i)\b(choice\s+of|choose\s+from|comes\s+with\s+choice\s+of)\s+(?P<opts>[^.;\n/]+)")

def split_option_list(s: str) -> List[str]:
    parts = re.split(r"\s*(?:or|and|/|,|\+)\s*", s.strip(), flags=re.IGNORECASE)
    return [p.strip(" -–—:()") for p in parts if p.strip(" -–—:()")]

def fallback_extract_modifiers(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    groups: Dict[str, List[Tuple[str, Optional[float]]]] = {}

    for m in MODIFIER_HEADER_RE.finditer(text):
        gcap = m.group(1).upper().strip()
        rest = text[m.end():]
        seg = re.split(r"(?:(?:\.\s)|\n{1,2})", rest, maxsplit=1)[0]
        tokens = re.split(r"[,\n;•/]+", seg)
        for tk in tokens:
            t = tk.strip()
            if not t:
                continue
            mm = PLUS_PRICE_LINE.match(t)
            if mm:
                nm = mm.group("name").strip()
                pr = smart_float(mm.group("price"))
            else:
                nm = t
                pr = None
            groups.setdefault(gcap, []).append((nm, pr))

    # Add patterns like "add X +1.0"
    for m in ADD_PATTERN.finditer(text):
        opts = split_option_list(m.group("opts"))
        pr = smart_float(m.group("price"))
        for o in opts:
            groups.setdefault("ADD", []).append((o, pr))

    # Patterns like "choice of fries, salad or rice"
    for m in CHOICE_PATTERN.finditer(text):
        opts = split_option_list(m.group("opts"))
        for o in opts:
            groups.setdefault("CHOICE", []).append((o, None))

    out = []
    for gcap, items in groups.items():
        out.append({
            "caption": gcap,
            "min": 0,
            "max": 1,
            "options": [{"caption": n, "price": p} for (n, p) in items]
        })
    return out

# ============================== Core: map extracted -> Flipdish JSON ==============================

def to_flipdish_json(
    extracted_pages: List[Dict[str, Any]],
    menu_name: str,
    price_band_id: str,
    attach_pdf_images: bool,
    src_pdf_doc: Optional["fitz.Document"],
    rules: Optional[dict] = None
) -> Dict[str, Any]:
    nowz = now_iso_z()
    out = {
        "etag": f"W/\"datetime'{nowz}'\"",
        "timestamp": now_iso_hms(),
        "id": guid(),
        "menuEditor": "v2",
        "name": smart_title(menu_name or "Generated Menu"),
        "isEnabled": True,
        "isPublished": False,
        "categories": [],
        "modifiers": [],
        "taxRates": [],
        "priceBands": [{
            "id": price_band_id,
            "caption": "Default",
            "isDefault": True
        }]
    }

    modifiers_index: Dict[str, Dict[str, Any]] = {}
    cat_index: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def _ensure_modifier_group(key: str, caption: str, min_sel: Optional[int], max_sel: Optional[int]) -> Dict[str, Any]:
        if key in modifiers_index:
            return modifiers_index[key]
        g = {
            "etag": f"W/\"datetime'{nowz}'\"",
            "timestamp": now_iso_hms(),
            "caption": caption,
            "enabled": True,
            "hiddenInOrderFlow": False,
            "id": guid(),
            "max": 1 if max_sel is None else int(max_sel),
            "min": 0 if min_sel is None else int(min_sel),
            "position": len(modifiers_index),
            "items": [],
            "overrides": []
        }
        g["_items_map"] = {}
        modifiers_index[key] = g
        return g

    def _ensure_option_item(group: Dict[str, Any], oname: str, price: Optional[float]) -> Dict[str, Any]:
        key_nm = oname.lower() + "|" + str(price if price is not None else -1)
        if key_nm in group["_items_map"]:
            return group["_items_map"][key_nm]
        nowh = now_iso_hms()

        detected_allergens = _detect_allergens_from_text(oname)

        opt_item = {
            "etag": f"W/\"datetime'{nowz}'\"",
            "timestamp": nowh,
            "autoSelectedQuantity": 0,
            "caption": oname,
            "doesPriceRepresentRewardPoints": False,
            "enabled": True,
            "id": guid(),
            "pricingProfiles": [{
                "etag": f"W/\"datetime'{nowz}'\"",
                "timestamp": nowh,
                "priceBandId": price_band_id,
                "collectionPrice": float(price or 0.0),
                "deliveryPrice": float(price or 0.0),
                "dineInPrice": float(price or 0.0),
                "takeawayPrice": float(price or 0.0),
            }],
            "charges": [],
            "modifierMembers": [],
            "overrides": []
        }

        if detected_allergens:
            opt_item["paramsJson"] = _attach_allergens_to_params(
                opt_item.get("paramsJson", ""),
                detected_allergens
            )

        group["_items_map"][key_nm] = opt_item
        return opt_item

    def _link_group_to_parent(parent_entity: Dict[str, Any], group_obj: Dict[str, Any]) -> None:
        parent_entity.setdefault("modifierMembers", [])
        parent_entity["modifierMembers"].append({
            "etag": f"W/\"datetime'{nowz}'\"",
            "timestamp": now_iso_hms(),
            "canSameItemBeSelectedMultipleTimes": group_obj.get("canSameItemBeSelectedMultipleTimes", True),
            "caption": group_obj["caption"],
            "id": group_obj["id"],
            "max": group_obj["max"],
            "min": group_obj["min"],
            "position": len(parent_entity["modifierMembers"]),
        })

    def _process_group(parent_entity: Dict[str, Any], grp: Dict[str, Any]) -> None:
        gcap = smart_title(grp.get("caption") or "Choose")
        min_sel = grp.get("min")
        max_sel = grp.get("max")
        key = f"{gcap}|{min_sel}|{max_sel}"
        g = _ensure_modifier_group(key, gcap, min_sel, max_sel)

        for opt in (grp.get("options") or []):
            oname = smart_title(opt.get("caption") or "")
            if not oname:
                continue
            price = opt.get("price")
            oi = _ensure_option_item(g, oname, price)
            # nothing else here; oi already has pricing and allergens, etc.

        _link_group_to_parent(parent_entity, g)

    # build categories & items
    for page_i, page_data in enumerate(extracted_pages):
        page_num = page_data.get("number", page_i + 1)
        for cat_in in page_data.get("categories", []):
            cname = smart_title(cat_in.get("caption") or f"Page {page_num}")
            pos = len(out["categories"])
            ck = (cname, page_num)
            if ck not in cat_index:
                color = infer_category_color(cname)
                cat = {
                    "etag": f"W/\"datetime'{nowz}'\"",
                    "timestamp": now_iso_hms(),
                    "id": guid(),
                    "caption": cname,
                    "enabled": True,
                    "hidden": False,
                    "items": [],
                    "position": pos,
                }
                if color:
                    cat["color"] = color
                out["categories"].append(cat)
                cat_index[ck] = cat
            else:
                cat = cat_index[ck]

            page = src_pdf_doc[page_i] if (attach_pdf_images and src_pdf_doc is not None) else None

            for it in (cat_in.get("items") or []):
                raw = (it.get("caption") or "").strip()
                desc = (it.get("description") or "").strip()
                notes = (it.get("notes") or "").strip()
                name, inline = split_caption_and_inline_notes(raw)
                name = smart_title(name or "Item")

                base_price = it.get("price")
                if base_price is None:
                    base_price = parse_price_from_text(raw, desc, notes) or 0.0

                img_data_url = ""
                if page is not None and name and fitz is not None:
                    rects = find_item_rects(page, name)
                    if rects:
                        png = nearest_image_crop(page, rects[0])
                        if png:
                            img_data_url = to_data_url(png)

                item = {
                    "etag": f"W/\"datetime'{nowz}'\"",
                    "timestamp": now_iso_hms(),
                    "caption": name,
                    "notes": " ".join(p for p in [desc, notes, inline] if p).strip(),
                    "enabled": True,
                    "id": guid(),
                    "doesPriceRepresentRewardPoints": False,
                    "pricingProfiles": [{
                        "etag": f"W/\"datetime'{nowz}'\"",
                        "timestamp": now_iso_hms(),
                        "priceBandId": price_band_id,
                        "collectionPrice": float(base_price),
                        "deliveryPrice": float(base_price),
                        "dineInPrice": float(base_price),
                        "takeawayPrice": float(base_price),
                    }],
                    "charges": [],
                    "modifierMembers": [],
                    "overrides": [],
                    "imageUrl": img_data_url or ""
                }

                detected_allergens = _detect_allergens_from_text(
                    name,
                    desc,
                    notes,
                    inline
                )
                if detected_allergens:
                    item["paramsJson"] = _attach_allergens_to_params(
                        item.get("paramsJson", ""),
                        detected_allergens
                    )

                llm_mods = it.get("modifiers") or []
                if not llm_mods:
                    llm_mods = fallback_extract_modifiers("\n".join([desc, notes, inline]))

                for grp in llm_mods:
                    _process_group(item, grp)

                cat["items"].append(item)

    # finalize modifiers
    for g in modifiers_index.values():
        g["items"] = list(g["_items_map"].values())
        del g["_items_map"]
        out["modifiers"].append(g)

    out["categories"] = [c for c in out["categories"] if c.get("items")]

    # Apply normalization rules silently (rules.json if present; defaults otherwise)
    rules = rules or try_load_rules()
    out = normalize_with_rules(out, rules)

    return out

def normalize_with_rules(flipdish_json: dict, rules: dict) -> dict:
    if not rules:
        return flipdish_json

    # Example: apply modifier_caption_aliases & force_minmax
    cap_alias = rules.get("modifier_caption_aliases") or {}
    force_mm = rules.get("force_minmax") or {}

    name_to_mod = {m["caption"].upper(): m for m in flipdish_json.get("modifiers", [])}

    # Aliases
    for canon, alist in cap_alias.items():
        canon_u = canon.upper()
        for alias in alist:
            mu = name_to_mod.get(alias.upper())
            if mu:
                mu["caption"] = canon
                name_to_mod[canon_u] = mu

    # Min/max
    for cap, mm in force_mm.items():
        mu = name_to_mod.get(cap.upper())
        if mu and isinstance(mm, (list, tuple)) and len(mm) == 2:
            mu["min"], mu["max"] = int(mm[0]), int(mm[1])

    return flipdish_json

# ============================== OpenAI extraction ==============================

SYSTEM_PROMPT = """You are an expert at structuring restaurant menus for Flipdish.
You will receive OCR text (already roughly grouped by layout).
Your job:
- Identify categories.
- Within each category, identify items.
- For each item, capture: caption, description (optional), price (if next to caption), notes (extras), and any obvious modifiers.
- If something looks like choices/add-ons/extras, put them into 'modifiers' for that item using a consistent structure.

Return JSON in this shape:
[
  {
    "categories": [
      {
        "caption": "Category Name",
        "items": [
          {
            "caption": "Item Name",
            "description": "Optional description",
            "price": 12.5,
            "notes": "Optional trailing notes",
            "modifiers": [
              {
                "caption": "Choose Side",
                "min": 0,
                "max": 1,
                "options": [
                  { "caption": "Fries", "price": 0.0 },
                  { "caption": "Salad", "price": 0.0 }
                ]
              }
            ]
          }
        ]
      }
    ]
  }
]

Be concise. Use numbers (not strings) for prices. If uncertain, omit the price.
"""

def call_openai_vision_extract(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if client is None:
        raise RuntimeError("OpenAI client not configured.")
    # For brevity: we send one concatenated text prompt.
    full_text = []
    for p in pages:
        full_text.append(f"# Page {p['number']}\n{p['text']}")
    prompt = "\n\n".join(full_text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    txt = resp.choices[0].message.content
    try:
        data = json.loads(txt)
    except Exception:
        data = {"categories": []}
    # Ensure list-of-pages style
    if isinstance(data, dict):
        data = [data]
    return data

# ============================== Streamlit UI ==============================

st.title("Flipdish Menu Builder")

tab1, tab2 = st.tabs(["From PDF/Image (AI)", "Reshape Existing JSON"])

with tab1:
    st.subheader("Upload a menu image or PDF")
    uploaded = st.file_uploader(
        "Upload PDF or image",
        type=["pdf", "png", "jpg", "jpeg"],
        key="menu_upload"
    )
    col_l, col_r = st.columns([2, 1])

    with col_r:
        menu_name = st.text_input("Menu Name", value="My Restaurant Menu")
        price_band_id = st.text_input("Price Band ID", value="default")
        attach_images = st.checkbox("Attach cropped item images", value=False)
        run_btn = st.button("Generate Flipdish JSON", type="primary")

    with col_l:
        if uploaded is not None:
            if uploaded.type == "application/pdf":
                st.caption("Preview unavailable for multi-page PDF in sidebar.")
            else:
                try:
                    img = Image.open(uploaded)
                    st.image(img, caption="Uploaded menu", use_column_width=True)
                except Exception:
                    st.warning("Could not render image preview.")

    if run_btn:
        if not uploaded:
            st.error("Please upload a PDF/image first.")
            st.stop()
        if not price_band_id.strip():
            st.error("Price Band ID is required.")
            st.stop()

        file_bytes = uploaded.read()
        src_pdf = None
        pages_data = []

        if uploaded.type == "application/pdf":
            if fitz is None:
                st.error("PyMuPDF not installed on server; cannot parse PDF.")
                st.stop()
            src_pdf = fitz.open(stream=file_bytes, filetype="pdf")
            for i, page in enumerate(src_pdf):
                text = page.get_text("text")
                blocks = page.get_text("blocks")
                pages_data.append({
                    "number": i + 1,
                    "text": text,
                    "blocks": blocks,
                })
        else:
            # Single image: minimal handling
            if fitz is None:
                st.error("PyMuPDF not installed; image OCR stub not implemented here.")
                st.stop()
            # Could pass-through to OCR; for now treat as one big page of dummy text
            pages_data = [{
                "number": 1,
                "text": "",
                "blocks": [],
            }]

        st.info("Calling OpenAI to interpret menu layout...")
        extracted = call_openai_vision_extract(pages_data)

        rules = try_load_rules()
        flipdish_json = to_flipdish_json(
            extracted_pages=extracted,
            menu_name=menu_name,
            price_band_id=price_band_id.strip(),
            attach_pdf_images=attach_images,
            src_pdf_doc=src_pdf,
            rules=rules
        )

        st.success("Menu JSON generated with auto allergen tagging.")
        st.json(flipdish_json, expanded=False)
        st.download_button(
            "Download Flipdish JSON",
            data=json.dumps(flipdish_json, indent=2, ensure_ascii=False).encode(),
            file_name="flipdish_menu.json",
            mime="application/json"
        )

with tab2:
    st.subheader("Reshape existing Flipdish-style JSON")
    jf = st.file_uploader(
        "Upload existing JSON menu",
        type=["json"],
        key="json_upload"
    )
    menu_name2 = st.text_input("Menu Name (optional override)", value="")
    price_band_id2 = st.text_input("Price Band ID", value="default", key="pb2")
    reshape_btn = st.button("Reshape / Normalize", key="reshape_btn")

    if reshape_btn:
        if not jf:
            st.error("Upload a JSON file first.")
            st.stop()
        if not price_band_id2.strip():
            st.error("Price Band ID is required.")
            st.stop()

        raw = json.load(io.BytesIO(jf.read()))
        result = to_flipdish_json(
            [raw],
            menu_name2 or "",
            price_band_id2.strip(),
            False,
            None,
            rules=None
        )
        st.success("Re-shaped successfully (with allergen tagging on detected items/options).")
        st.json(result, expanded=False)
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
            file_name="flipdish_menu.json",
            mime="application/json"
        )
