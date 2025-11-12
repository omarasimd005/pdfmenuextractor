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
import hashlib
import math
from typing import List, Dict, Any, Optional, Tuple

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
# --- MODIFICATION: Disable the Decompression Bomb check ---
Image.MAX_IMAGE_PIXELS = None
# --- END MODIFICATION ---
load_dotenv()
st.set_page_config(page_title="Flipdish Menu Builder", page_icon="🍽️", layout="centered")
st.title("Flipdish Menu Builder")

# ============================== Utils ==============================

def guid() -> str:
    return str(uuid.uuid4())

def now_iso_hms() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def now_iso_z() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def to_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("utf-8")

# ============================== Loaders ==============================

def _is_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF")

def _open_image_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im.load()
    return im.convert("RGB")

class LoadedFile:
    def __init__(self, images: List[Image.Image], doc: Optional["fitz.Document"], is_pdf: bool):
        self.images, self.doc, self.is_pdf = images, doc, is_pdf

def load_file(file) -> LoadedFile:
    if file is None:
        return LoadedFile([], None, False)
    try:
        file.seek(0)
    except Exception:
        pass
    data = file.read()
    if not data:
        return LoadedFile([], None, False)

    name = (getattr(file, "name", "") or "").lower()
    if _is_pdf(data) or name.endswith(".pdf"):
        if fitz is None:
            return LoadedFile([], None, True)
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception:
            return LoadedFile([], None, True)
        pages = []
        try:
            for i in range(len(doc)):
                # --- MODIFICATION: Changed DPI back to 200 for better accuracy ---
                pix = doc[i].get_pixmap(dpi=200)
                pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        except Exception:
            return LoadedFile([], doc, True)
        return LoadedFile(pages, doc, True)

    try:
        img = _open_image_from_bytes(data)
        return LoadedFile([img], None, False)
    except Exception:
        return LoadedFile([], None, False)

# ============================== Text helpers ==============================

SMALL_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in",
    "into", "nor", "of", "on", "onto", "or", "per", "the", "to", "vs",
    "via", "with", "over", "under", "up", "down", "off"
}

# --- MODIFICATION: Split allergens into two groups based on user's screenshot ---

# Group 1: Official Flipdish tags for paramsJson (from screenshot)
FLIPDISH_OFFICIAL_ALLERGENS = {
    "celery": "Celery",
    "crustacean": "Crustaceans",
    "crustaceans": "Crustaceans",
    "egg": "Egg",
    "eggs": "Egg",
    "fish": "Fish",
    "gluten": "Gluten", # Note: "Gluten Free" is separate
    "lupin": "Lupin",
    "milk": "Milk",
    "lactose": "Milk", # Common alias
    "dairy": "Milk",   # Common alias
    "mollusc": "Molluscs",
    "molluscs": "Molluscs",
    "mustard": "Mustard",
    "nut": "Nuts",     # Covers various nuts
    "nuts": "Nuts",
    "peanut": "Peanuts",
    "peanuts": "Peanuts",
    "sesame": "Sesame",
    "soya": "Soya",
    "soy": "Soya",     # Common alias
    "soybean": "Soybeans",
    "soybeans": "Soybeans",
    "sulphur": "Sulphur Dioxide",
    "sulphur dioxide": "Sulphur Dioxide",
    "sulphites": "Sulphur Dioxide",
    "wheat": "Wheat",
    "alcohol": "Alcohol",
    "(alc)": "Alcohol"
}
FLIPDISH_OFFICIAL_LIST = sorted(list(set(FLIPDISH_OFFICIAL_ALLERGENS.values())))

# Group 2: Other tags for user-facing description (notes field)
OTHER_DIETARY_TAGS = {
    "gluten-free": "Gluten Free",
    "gluten free": "Gluten Free",
    "(gf)": "Gluten Free",
    # "g": "Gluten Free", # REMOVED - AI should find this from legend
    "vegan": "Vegan",
    "(ve)": "Vegan",
    "(v)": "Vegan", # From Birch Tree menu
    "vegetarian": "Vegetarian",
    "(vg)": "Vegetarian", # From Birch Tree menu
    "halal": "Halal",
    "(h)": "Halal",
    "spicy": "Spicy", # General spicy
    "mild": "Mild Spice",
    "mild spice": "Mild Spice",
    "medium": "Medium Spice",
    "medium spice": "Medium Spice",
    "hot": "Hot Spice",
    "hot spice": "Hot Spice",
    "extra hot": "Hot Spice",
    # "000": "Spicy", # REMOVED - AI should find this from legend
    # "00": "Spicy", # REMOVED - AI should find this from legend
    # "0": "Spicy", # REMOVED - AI should find this from legend
}
OTHER_DIETARY_LIST = sorted(list(set(OTHER_DIETARY_TAGS.values())))

# Combined dictionary for text scanning
ALL_DIETARY_KEYWORDS = {**FLIPDISH_OFFICIAL_ALLERGENS, **OTHER_DIETARY_TAGS}
# --- END MODIFICATION ---

# --- MODIFICATION: Added "quantity" keywords ---
SIZE_MODIFIER_PATTERNS = [
    # Size
    re.compile(r'\bsize\b', re.I),
    re.compile(r'\bsizes\b', re.I),
    re.compile(r'^select a size$', re.I),
    re.compile(r'^choose your size$', re.I),
    re.compile(r'^select size$', re.I),
    re.compile(r'^choose size$', re.I),
    # Quantity
    re.compile(r'\bquantity\b', re.I),
    re.compile(r'\bpieces\b', re.I), # e.g., "6 pieces", "12 pieces"
    re.compile(r'^select quantity$', re.I),
    re.compile(r'^choose quantity$', re.I),
    # Columns
    re.compile(r'\bsingle\b', re.I), # For "Single" vs "Meal"
    re.compile(r'\bmeal\b', re.I),
]
# --- END MODIFICATION ---


def _cap_hyphenated(token: str) -> str:
    return "-".join(p.capitalize() if p else p for p in token.split("-"))

def smart_title(text: str) -> str:
    if not text:
        return text
    tokens = re.split(r'(\s+)', text.strip())
    words_only = [t for t in tokens if not re.match(r'\s+', t)]
    result = []
    word_index = 0
    for t in tokens:
        if re.match(r'\s+', t):
            result.append(t); continue
        lower = t.lower()
        
        base = _cap_hyphenated(lower)
        if (word_index == 0 or word_index == len(words_only) - 1 or lower not in SMALL_WORDS):
            out = base[0].upper() + base[1:] if base else base
        else:
            out = base.lower()

        result.append(out); word_index += 1
    
    final_string = "".join(result)
    final_string = final_string.replace(" and ", " & ")
    final_string = final_string.replace(" And ", " & ")
    return final_string

# --- MODIFICATION: Changed from Regex list to simple dictionary lookup ---
CATEGORY_COLOR_MAP = {
    "Starters": {"backgroundColor": "#E67E22", "foregroundColor": "#FFFFFF"},
    "Mains": {"backgroundColor": "#C0392B", "foregroundColor": "#FFFFFF"},
    "Sides": {"backgroundColor": "#FBC02D", "foregroundColor": "#000000"},
    "Soup/Salad": {"backgroundColor": "#2E8B57", "foregroundColor": "#FFFFFF"},
    "Desserts": {"backgroundColor": "#8E44AD", "foregroundColor": "#FFFFFF"},
    "Drinks": {"backgroundColor": "#9b9b9b", "foregroundColor": "#FFFFFF"},
    "Specials": {"backgroundColor": "#FF66B2", "foregroundColor": "#000000"}, # Corrected foreground
    "Kids": {"backgroundColor": "#3498DB", "foregroundColor": "#FFFFFF"},
    "Pizza": {"backgroundColor": "#D64541", "foregroundColor": "#FFFFFF"},
    "Burgers": {"backgroundColor": "#935116", "foregroundColor": "#FFFFFF"},
    "Sauces": {"backgroundColor": "#FFDAB9", "foregroundColor": "#000000"},
    "Alcohol": {"backgroundColor": "#2C3E50", "foregroundColor": "#FFFFFF"},
    "Other": None, # Default, no color
}
CATEGORY_TYPES_LIST = sorted(list(CATEGORY_COLOR_MAP.keys()))
# --- END MODIFICATION ---


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _relative_luminance(rgb: Tuple[int, int, int]) -> float:
    def _lin(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (_lin(v) for v in rgb)
    return 0.2126*r + 0.7152*g + 0.0722*b

# --- MODIFICATION: Changed to simple dictionary lookup by type ---
def pick_category_colors(category_type: str) -> Optional[Dict[str, str]]:
    """
    Returns the color dict for a given category type.
    """
    if not category_type:
        return None
    # Find the type, default to "Other"
    return CATEGORY_COLOR_MAP.get(category_type, CATEGORY_COLOR_MAP["Other"])
# --- END MODIFICATION ---

def format_description(text: str) -> str:
    """
    Cleans up description text:
    - Replaces & with 'and'
    - Ensures it ends with a full stop.
    - Capitalizes the first letter of the string.
    - Capitalizes the first letter after a full stop.
    """
    text = (text or "").strip()
    if not text:
        return ""
    
    text = text.replace("&", "and")
    
    if text and text[-1] not in ".!?":
        text += "."
        
    if text:
        text = text[0].upper() + text[1:]
    
    text = re.sub(r'(?<=\.\s)(\w)', lambda m: m.group(1).upper(), text)
    
    return text

# --- NEW HELPER FUNCTION for size price logic ---
def _is_size_modifier_group(caption: str) -> bool:
    if not caption:
        return False
    cap_norm = caption.strip()
    for pat in SIZE_MODIFIER_PATTERNS:
        if pat.search(cap_norm):
            return True
    return False

def _normalize_modifier_prices(modifier_groups: list, item_base_price: float) -> Tuple[list, float]:
    """
    Checks for size-based modifiers with absolute prices.
    Updates the item's base price to be the minimum found.
    Recalculates modifier option prices as the difference.
    """
    if not modifier_groups:
        return modifier_groups, item_base_price

    new_base_price = item_base_price
    
    # First pass: find all absolute size prices and determine the true base price
    all_absolute_prices = []
    if item_base_price is not None and item_base_price > 0: # Only count if > 0
        all_absolute_prices.append(item_base_price)

    has_absolute_price_group = False
    for grp in modifier_groups:
        if _is_size_modifier_group(grp.get("caption")):
            options = grp.get("options", [])
            if not options:
                continue

            # Heuristic: If ANY option has a price > 0, assume it's an absolute price group.
            if any(opt.get("price") is not None and opt.get("price") > 0 for opt in options):
                has_absolute_price_group = True
                for opt in options:
                    if opt.get("price") is not None:
                        all_absolute_prices.append(opt["price"])

    # If we found absolute prices, set the new base price
    if has_absolute_price_group and all_absolute_prices:
        new_base_price = min(all_absolute_prices)
    elif not all_absolute_prices and item_base_price is None:
        # Edge case: No prices found anywhere, default to 0
        new_base_price = 0.0
    elif item_base_price is not None:
        # No absolute modifier prices, so the item's own price is the base
        new_base_price = item_base_price
    elif all_absolute_prices:
        # Item price was null, but we have modifier prices
        new_base_price = min(all_absolute_prices)
    else:
        # Default fallback
        new_base_price = 0.0


    # Second pass: Recalculate prices as differences from the new base price
    for grp in modifier_groups:
        if _is_size_modifier_group(grp.get("caption")):
            options = grp.get("options", [])
            if not options:
                continue

            # Check again if this group was identified as absolute (safe to re-check)
            if any(opt.get("price") is not None and opt.get("price") > 0 for opt in options):
                for opt in options:
                    if opt.get("price") is not None:
                        opt["price"] = max(0.0, opt["price"] - new_base_price)
                    else:
                        # If a size option has no price (e.g. "Small"), assume it IS the base price
                        opt["price"] = 0.0

    return modifier_groups, new_base_price
# --- END NEW HELPER FUNCTIONS ---


# ============================== Price parsing ==============================

PRICE_RE = re.compile(r'(?:£|\$|€)?\s*(\d{1,3}(?:\.\d{1,2})?)')
PLUS_PRICE_RE = re.compile(r'(?i)^(?P<name>.*?\S)\s*(?:\+|plus\s*)(?P<price>\d+(?:\.\d+)?)\s*$')
PENCE_RE = re.compile(r'(\d{1,3})\s*(?:p|P)\b')

def parse_price_from_text(*texts: str) -> Optional[float]:
    for t in texts or []:
        if not t: continue
        m = PRICE_RE.search(t)
        if m:
            try: return float(m.group(1))
            except Exception: pass
        if PENCE_RE.search(t): continue
    return None

_SPLIT_PATTERNS = [
    re.compile(r"^(?P<name>.+?)\s*[-–—:]\s*(?P<desc>.+)$"),
    re.compile(r"^(?P<name>.+?)\s*\((?P<desc>[^)]+)\)\s*$"),
]
def split_caption_and_inline_notes(text: str) -> Tuple[str, str]:
    t = (text or "").strip()
    for p in _SPLIT_PATTERNS:
        m = p.match(t)
        if m: return m.group("name").strip(), m.group("desc").strip()
    return t, ""

# ============================== Learning store ==============================

EXAMPLES_PATH = "examples.jsonl"  # newline-delimited examples
DEFAULT_RULES = {
    "modifier_caption_aliases": {
        "ADD": ["EXTRAS", "ADD-ONS", "GOES WELL WITH", "SIDES"],
        "CHOOSE PROTEIN": ["PROTEIN CHOICE", "CHOOSE YOUR PROTEIN"]
    },
    "force_minmax": {
        "CHOOSE PROTEIN": {"min": 1, "max": 1},
        "ADD": {"min": 0, "max": 5}
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
                if not line: continue
                try: out.append(json.loads(line))
                except Exception: pass
    except Exception:
        return []
    return out

def _tokenize(t): 
    return re.findall(r"[a-z0-9]+", (t or "").lower())

def _bow(text):
    d = {}
    for tok in _tokenize(text):
        d[tok] = d.get(tok, 0) + 1
    return d

def _cos(a,b):
    num = sum(a.get(k,0)*b.get(k,0) for k in set(a)|set(b))
    den = math.sqrt(sum(v*v for v in a.values()))*math.sqrt(sum(v*v for v in b.values()))
    return num/den if den else 0.0

def top_k_examples(query_text, k=3):
    q = _bow(query_text or "")
    if not q: return []
    exs = load_examples()
    scored = []
    for ex in exs:
        s = _cos(q, _bow(ex.get("source","")))
        if s > 0: scored.append((s, ex))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [ex for _, ex in scored[:k]]

def build_fewshot_context(query_text: str) -> str:
    shots = top_k_examples(query_text, k=3)
    if not shots: return ""
    lines = []
    for ex in shots:
        lines.append(json.dumps({
            "source_excerpt": (ex.get("source","") or "")[:400],
            "expected_flipdish_piece": ex.get("flipdish",{})
        }, ensure_ascii=False))
    return "\n".join(lines)

def try_load_rules() -> dict:
    # auto-load rules.json from app root if present; otherwise defaults
    path = "rules.json"
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_RULES
    return DEFAULT_RULES

# ============================== Vision extraction ==============================

# --- MODIFICATION: Added "Master Price Block" and "Strict Sectional Boundaries" rules ---
BASE_EXTRACTION_PROMPT = f"""
You output ONLY JSON (no markdown) with this schema:

{{
  "name": string,
  "categories": [
    {{
      "caption": string,
      "description": string,
      "category_type": string,
      "items": [
        {{
          "caption": string,
          "description": string,
          "notes": string,
          "price": number,
          "official_allergens": [string],
          "other_dietary_tags": [string],
          "modifiers": [
            {{
              "caption": string,
              "min": number|null,
              "max": number|null,
              "options": [
                {{
                  "caption": string,
                  "price": number|null,
                  "modifiers": [
                    {{
                      "caption": string,
                      "min": number|null,
                      "max": number|null,
                      "options": [{{"caption": string, "price": number|null}}]
                    }}
                  ]
                }}
              ]
            }}
          ]
        }}
      ]
    }}
  ]
}}

Detect modifiers from many phrasings:
- "GOES WELL WITH X +Y, Z +W"
- "Add ham +3", "Add chicken or chorizo +4", "Add steak +6"
- "Choice of toast or pancakes", "Choose from ..."
- Upgrades: "… (+2 to upgrade …)" as options with price
- CONDITIONALS: If a choice leads to another selection (e.g., size -> sides), attach the follow-up group(s) under that option's "modifiers".

Rules:
- **Step 1: Find the Legend:** First, scan the *entire* menu to find any "Allergy Key" or "Legend" that defines symbols (e.g., 🌶️ = Spicy, (G) = Gluten-Free, (V) = Vegetarian). Remember this legend.
- **Step 2: Apply the Legend:** As you extract each item, use the legend you found. If an item has a symbol (like 🌶️, 000, or G), add the corresponding tag (e.g., "Spicy" or "Gluten Free") to its `other_dietary_tags` or `official_allergens` list.
- **Step 3: Find Other Tags:** Also look for simple text or common symbols (e.g., (N), (H), (Alc)) indicating allergens, alcohol, Vegan, Vegetarian, Halal, or Spice Levels, even if they aren't in the legend.
- **Master Price Blocks:** If a category (e.g., "Pizza") lists a price grid at the TOP (e.g., "Small $10, Medium $12, Large $14") and the items below it ("Margherita", "Pepperoni") have NO prices, you MUST:
    1.  Create a "Size" or "Type" modifier group for *each* item ("Margherita", "Pepperoni").
    2.  Add options to that group using the prices from the master grid (e.g., "Small $10", "Medium $12", "Large $14").
    3.  Set the main `price` for the item itself to `null` or `0`.
- **Strict Sectional Boundaries:** Pay close attention to headings. Do not mix items from one section (e.g., "Normal Burgers") with another section (e.g., "Special Burgers"). A new heading must start a new, distinct category.
- Item price numeric; ignore currency symbols.
- Options without explicit price -> price=null.
- Keep headings with a price as items; ignore decorative section headers.
- **Special Offers:** If you find any deals, bundles, or special offers (e.g., "Family Meal," "Lunch Special," "2-for-1 Deal"), group them as items under a new category named "Special Offers".
- Populate `official_allergens` ONLY with tags from this list: {json.dumps(FLIPDISH_OFFICIAL_LIST)}
- Populate `other_dietary_tags` with tags like "Vegan", "Vegetarian", "Halal", "Gluten Free", or spice levels from this list: {json.dumps(OTHER_DIETARY_LIST)}
- **Strict Boundaries:** Each item is distinct. Be very careful to only associate modifiers, prices, and descriptions that are clearly and closely related to a single item. Do not 'mix' or 'bleed' information (like prices or add-ons) from one item to another.
- **Category Type:** For each category, classify its `caption` into one of these types: {json.dumps(CATEGORY_TYPES_LIST)}. If the category is "Tea", "Coffee", or "Hot Drinks", classify it as "Drinks". If "Wine" or "Spirits", classify as "Alcohol".
"""
# --- END MODIFICATION ---

def _img_hash(img: Image.Image) -> str:
    return hashlib.blake2b(img.tobytes(), digest_size=16).hexdigest()

@st.cache_data(show_spinner=False, ttl=7*24*3600)
def _cached_extract_page(img_bytes: bytes, model: str, fewshot: str) -> Dict[str, Any]:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return _run_openai_single_uncached(im, model=model, fewshot=fewshot)

def run_openai_single(img: Image.Image, model: str = "gpt-4o", fewshot: str = "") -> Dict[str, Any]:
    buf = io.BytesIO(); img.save(buf, "PNG")
    return _cached_extract_page(buf.getvalue(), model, fewshot)

def _run_openai_single_uncached(image: Image.Image, model: str = "gpt-4o", fewshot: str = "") -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        # Minimal offline stub (so UI doesn’t break)
        return {
            "name": "Sample",
            "categories": [{
                "caption": "brunch",
                "description": "Our famous brunch selection",
                "category_type": "Mains",
                "items": [{
                    "caption": "Combo Breakfast",
                    "description": "Choose main; if Waffles then choose syrup. (V), (Contains Gluten, Egg, Milk)",
                    "price": 28,
                    "official_allergens": ["Gluten", "Egg", "Milk"],
                    "other_dietary_tags": ["Vegetarian"],
                    "modifiers": [{
                        "caption": "Choose Main",
                        "min": 1, "max": 1,
                        "options": [
                            {"caption": "Pancakes", "price": None},
                            {"caption": "Waffles", "price": None, "modifiers": [{
                                "caption": "Choose Syrup", "min": 1, "max": 1,
                                "options": [{"caption": "Maple", "price": None}, {"caption": "Chocolate", "price": None}]
                            }]}
                        ]
                    }]
                }]}
            ]
        }

    sys_prompt = BASE_EXTRACTION_PROMPT
    if fewshot:
        sys_prompt += "\n\nEXAMPLES (follow this structure and style):\n" + fewshot

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract menu JSON with explicit and conditional modifiers for this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(image)}"}}
            ]}
        ],
    )
    return json.loads(resp.choices[0].message.content)

# ============================== PDF helpers (optional images) ==============================

def find_item_rects(page: "fitz.Page", name_clean: str) -> List["fitz.Rect"]:
    if fitz is None or not name_clean.strip():
        return []
    r = page.search_for(name_clean)
    if r: return r
    toks = name_clean.split()
    for n in (3, 2, 1):
        if len(toks) >= n:
            r = page.search_for(" ".join(toks[:n]))
            if r: return r
    return []

def nearest_image_crop(page: "fitz.Page", near: "fitz.Rect", margin: float = 12.0) -> Optional[bytes]:
    if fitz is None:
        return None
    layout = page.get_text("dict")
    imgs = []
    for b in layout.get("blocks", []):
        if b.get("type") == 1 and "bbox" in b:
            x0, y0, x1, y1 = b["bbox"]
            imgs.append(fitz.Rect(x0, y0, x1, y1))
    if not imgs:
        return None
    best, best_d = None, 1e9
    for ir in imgs:
        ay = (near.y0 + near.y1) / 2
        iy = (ir.y0 + ir.y1) / 2
        d = abs(ay - iy)
        if d < best_d:
            best, best_d = ir, d
    if not best: return None
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
    if not text: return []
    groups: Dict[str, List[Tuple[str, Optional[float]]]] = {}

    for m in MODIFIER_HEADER_RE.finditer(text):
        gcap = m.group(1).upper().strip()
        rest = text[m.end():]
        seg = re.split(r"(?:(?:\.\s)|\n{1,2})", rest, maxsplit=1)[0]
        tokens = re.split(r"[,\n;•/]+", seg)
        for tk in tokens:
            t = tk.strip()
            if not t: continue
            if t.isupper() and len(t.split()) <= 5 and not PRICE_RE.search(t):
                break
            pm = PLUS_PRICE_LINE.match(t)
            if pm:
                groups.setdefault(gcap, []).append((pm.group("name").strip(" -–—:"), float(pm.group("price"))))
            else:
                groups.setdefault(gcap, []).append((t.strip(" -–—:"), None))

    for m in ADD_PATTERN.finditer(text):
        names = split_option_list(m.group("opts"))
        price = float(m.group("price")) if m.group("price") else None
        if names:
            groups.setdefault("ADD", [])
            for n in names:
                groups["ADD"].append((n, price))

    for line in re.split(r"[.;\n]+", text):
        pm = PLUS_PRICE_LINE.match(line.strip())
        if pm:
            groups.setdefault("ADD", []).append((pm.group("name").strip(" -–—:"), float(pm.group("price"))))

    for m in CHOICE_PATTERN.finditer(text):
        names = split_option_list(m.group("opts"))
        if names:
            groups.setdefault("CHOICE OF", [])
            for n in names:
                groups["CHOICE OF"].append((n, 0.0))

    out = []
    for cap, items in groups.items():
        seen, opts = set(), []
        for n, p in items:
            key = (n.lower(), p if p is not None else -1)
            if key in seen: continue
            seen.add(key)
            opts.append({"caption": n, "price": p})
        if opts:
            out.append({"caption": cap, "min": None, "max": None, "options": opts})
    return out

# ============================== Flipdish builder (conditional modifiers) ==============================

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
        "revisionId": "1",
        "status": "Draft",
        "type": "Store",
        "categories": [],
        "modifiers": [],
        "categoryGroups": []
    }

    modifiers_index: Dict[str, Dict[str, Any]] = {}

    def ensure_group(caption: str, min_sel: Optional[int] = None, max_sel: Optional[int] = None, can_repeat: Optional[bool] = None) -> Dict[str, Any]:
        key_raw = caption or "ADD"
        caption_smart = smart_title(key_raw if key_raw else "ADD")
        key = caption_smart.strip().upper() 
        
        nowh = now_iso_hms()
        if key in modifiers_index:
            return modifiers_index[key]
        g = {
            "etag": f"W/\"datetime'{nowz}'\"",
            "timestamp": nowh,
            "canSameItemBeSelectedMultipleTimes": True if can_repeat is None else bool(can_repeat),
            "caption": caption_smart, 
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
        group["_items_map"][key_nm] = opt_item
        return opt_item

    def _link_group_to_parent(parent_entity: Dict[str, Any], group_obj: Dict[str, Any]) -> None:
        parent_entity.setdefault("modifierMembers", [])
        parent_entity["modifierMembers"].append({
            "etag": f"W/\"datetime'{nowz}'\"",
            "timestamp": now_iso_hms(),
            "canSameItemBeSelectedMultipleTimes": group_obj.get("canSameItemBeSelectedMultipleTimes", True),
            "caption": group_obj["caption"], 
            "id": guid(),
            "max": group_obj["max"],
            "min": group_obj["min"],
            "modifierId": group_obj["id"]
        })

    def _process_group(parent_entity: Dict[str, Any], grp: Dict[str, Any]) -> None:
        if not grp: return
        g_caption = (grp.get("caption") or "ADD") 
        g_min = grp.get("min"); g_max = grp.get("max")
        can_repeat = grp.get("canSameItemBeSelectedMultipleTimes")
        group = ensure_group(g_caption, g_min, g_max, can_repeat) 

        for opt in (grp.get("options") or []):
            oname = smart_title((opt.get("caption") or "").strip()) 
            if not oname: continue
            price = opt.get("price")
            opt_item = _ensure_option_item(group, oname, price)
            for child_grp in (opt.get("modifiers") or []):
                _process_group(opt_item, child_grp)

        _link_group_to_parent(parent_entity, group)

    cat_index: Dict[str, Any] = {}
    
    last_good_category_key: Optional[str] = None

    for page_i, data in enumerate(extracted_pages):
        for cat_in in (data.get("categories") or []):
            cat_caption_raw = (cat_in.get("caption") or "Category").strip()
            
            is_generic = (cat_in.get("caption") is None or cat_caption_raw.lower() == "category")
            
            if is_generic and last_good_category_key:
                cat = cat_index[last_good_category_key]
            else:
                cat_caption = smart_title(cat_caption_raw).upper()
                cat_caption = re.sub(r'\bAND\b', '&', cat_caption)
                cat_description = format_description(cat_in.get("description"))
                
                # --- MODIFICATION: Read new category_type field from AI ---
                cat_type = cat_in.get("category_type", "Other") # Default to "Other"
                ck = cat_caption.lower()
                
                if ck not in cat_index:
                    cat = {
                        "etag": f"W/\"datetime'{nowz}'\"",
                        "timestamp": now_iso_hms(),
                        "caption": cat_caption,
                        "notes": cat_description, 
                        "enabled": True,
                        "id": guid(),
                        "items": [],
                        "overrides": []
                    }
                    # Use the *classified type* for colors, not the name
                    colors = pick_category_colors(cat_type)
                    if colors:
                        cat["backgroundColor"] = colors["backgroundColor"]
                        cat["foregroundColor"] = colors["foregroundColor"]
                    out["categories"].append(cat)
                    cat_index[ck] = cat
                else:
                    cat = cat_index[ck] 
                    if cat_description:
                        cat["notes"] = cat_description
                
                last_good_category_key = ck
            # --- END MODIFICATION ---

            page = src_pdf_doc[page_i] if (attach_pdf_images and src_pdf_doc is not None) else None

            # --- MODIFICATION: Re-ordered logic to support size price calculation ---
            for it in (cat_in.get("items") or []):
                
                # 1. Get all text and name info first
                raw = (it.get("caption") or "").strip()
                desc = (it.get("description") or "").strip()
                notes = (it.get("notes") or "").strip()
                name, inline = split_caption_and_inline_notes(raw)
                name = smart_title(name or "Item") 
                raw_item_notes = " ".join(p for p in [desc, notes, inline] if p).strip()

                # 2. Get initial base price
                base_price = it.get("price")
                # if base_price is None: # Old logic
                #     base_price = parse_price_from_text(raw, desc, notes) or 0.0

                # 3. Get modifiers
                llm_mods = it.get("modifiers") or []
                if not llm_mods:
                    llm_mods = fallback_extract_modifiers(raw_item_notes)
                
                # 4. === NEW: Normalize prices and find TRUE base price ===
                # This will find the lowest price, even if item price is 0 or null
                llm_mods, base_price = _normalize_modifier_prices(llm_mods, base_price)
                
                # 5. Get image URL
                img_data_url = ""
                if page is not None and name and fitz is not None:
                    rects = find_item_rects(page, name)
                    if rects:
                        png = nearest_image_crop(page, rects[0])
                        if png: img_data_url = to_data_url(png)

                # 6. Process dietary tags
                params_json_obj = {}
                ai_official_tags = it.get("official_allergens") or []
                ai_other_tags = it.get("other_dietary_tags") or []
                all_detected_tags = set(ai_official_tags) | set(ai_other_tags)
                
                scan_text = (raw_item_notes.lower() + " " + raw.lower())
                abbreviations = re.findall(r'(\([\s]*[A-Z]{1,3}[\s]*\))', raw_item_notes, re.I)
                abbreviations += re.findall(r'(\([\s]*[A-Z]{1,3}[\s]*\))', raw, re.I) # Check raw title too
                scan_text += " " + " ".join(abbreviations).lower()

                # --- MODIFICATION: Scan for custom legend symbols too ---
                # This looks for G, 0, 00, 000 etc. but NOT as part of another word
                # We search the raw caption for these symbols
                custom_symbols = re.findall(r'\b([G0]{1,5})\b', raw) 
                scan_text += " " + " ".join(custom_symbols).lower()
                # --- END MODIFICATION ---

                for keyword, flipdish_tag in ALL_DIETARY_KEYWORDS.items():
                    # Check for keyword OR abbreviation
                    if re.search(fr'\b{re.escape(keyword)}\b', scan_text, re.I):
                         all_detected_tags.add(flipdish_tag)
                
                final_official_tags = sorted(list(set(
                    [tag for tag in all_detected_tags if tag in FLIPDISH_OFFICIAL_LIST]
                )))
                
                final_other_tags = sorted(list(set(
                    [tag for tag in all_detected_tags if tag in OTHER_DIETARY_LIST]
                )))

                if final_official_tags:
                    params_json_obj["dietaryConfiguration"] = {
                        "dietaryTags": ",".join(final_official_tags)
                    }
                    
                all_tags_for_description = sorted(list(set(final_official_tags) | set(final_other_tags)))
                
                temp_raw_notes = raw_item_notes # Use a temp var
                if all_tags_for_description:
                    tag_string = f"(Contains: {', '.join(all_tags_for_description)})"
                    if temp_raw_notes and temp_raw_notes[-1] not in ".!?":
                        temp_raw_notes += ". " + tag_string
                    else:
                        temp_raw_notes += " " + tag_string
                
                item_notes = format_description(temp_raw_notes)
                
                # 7. Create the item dictionary (NOW uses the corrected base_price)
                item = {
                    "etag": f"W/\"datetime'{nowz}'\"",
                    "timestamp": now_iso_hms(),
                    "caption": name,
                    "notes": item_notes, # Apply formatted description
                    "enabled": True,
                    "id": guid(),
                    "doesPriceRepresentRewardPoints": False,
                    "pricingProfiles": [{
                        "etag": f"W/\"datetime'{nowz}'\"",
                        "timestamp": now_iso_hms(),
                        "priceBandId": price_band_id,
                        "collectionPrice": float(base_price), # USES CORRECTED BASE PRICE
                        "deliveryPrice": float(base_price), # USES CORRECTTED BASE PRICE
                        "dineInPrice": float(base_price), # USES CORRECTED BASE PRICE
                        "takeawayPrice": float(base_price), # USES CORRECTED BASE PRICE
                    }],
                    "charges": [],
                    "modifierMembers": [],
                    "overrides": [],
                    "imageUrl": img_data_url or "",
                    "paramsJson": json.dumps(params_json_obj) if params_json_obj else "{}"
                }

                # 8. Process the modifier groups (NOW uses corrected prices)
                for grp in llm_mods:
                    _process_group(item, grp)

                cat["items"].append(item)
            # --- END MODIFICATION ---

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
    aliases = rules.get("modifier_caption_aliases", {})
    force = rules.get("force_minmax", {})
    opt_alias = rules.get("option_aliases", {})

    def canon_mod_name(name):
        n = (name or "").strip().upper() 
        for target, alist in aliases.items():
            if n == target or n in [a.upper() for a in alist]:
                return target.upper() 
        return n or "ADD"

    for g in flipdish_json.get("modifiers", []):
        caption_key = canon_mod_name(g.get("caption"))
        
        if caption_key != g.get("caption", "").upper() and caption_key in aliases:
             g["caption"] = smart_title(caption_key)
        
        if caption_key in force: 
            mm = force[caption_key]
            if "min" in mm: g["min"] = int(mm["min"])
            if "max" in mm: g["max"] = int(mm["max"])
            
        for it in g.get("items", []):
            label = (it.get("caption","") or "").strip().lower()
            for canon, alist in opt_alias.items():
                if label in [canon] + alist:
                    it["caption"] = smart_title(canon) 
    return flipdish_json

# --- NEW QA FUNCTION ---
def run_qa_checks(flipdish_json: dict) -> List[Dict[str, str]]:
    """
    Scans the final JSON for common issues and returns a list of warnings.
    """
    warnings = []
    cat_names = []
    mod_names = []

    # Check categories and items
    for cat in flipdish_json.get("categories", []):
        cat_name = cat.get("caption", "Unnamed Category")
        if cat_name.lower() == "category" or cat_name == "Unnamed Category":
            warnings.append({"Type": "Category", "Name": "N/A", "Issue": "A category is missing a proper name."})
        
        cat_names.append(cat_name)

        if not cat.get("items"):
            warnings.append({"Type": "Category", "Name": cat_name, "Issue": "Category has 0 items."})
        
        for item in cat.get("items", []):
            item_name = item.get("caption", "Unnamed Item")
            if item_name.lower() == "item" or item_name == "Unnamed Item":
                warnings.append({"Type": "Item", "Name": "N/A", "Issue": f"An item in category '{cat_name}' is missing a name."})
            
            try:
                price = item.get("pricingProfiles", [{}])[0].get("collectionPrice", 0.0)
                # Warn if price is 0 AND it's not a container for modifiers
                if price == 0.0 and not item.get("modifierMembers"):
                     warnings.append({"Type": "Price", "Name": item_name, "Issue": "Item has a price of 0.00 and no modifiers."})
            except Exception:
                 warnings.append({"Type": "Price", "Name": item_name, "Issue": "Item has a missing or invalid price profile."})

    # Check modifier groups
    for mod in flipdish_json.get("modifiers", []):
        mod_name = mod.get("caption", "Unnamed Modifier")
        if mod_name.lower() == "add" or mod_name == "Unnamed Modifier":
             warnings.append({"Type": "Modifier", "Name": "N/A", "Issue": "A modifier group is missing a proper name (e.g., 'ADD')."})
        
        mod_names.append(mod_name)
        
        if not mod.get("items"):
            warnings.append({"Type": "Modifier", "Name": mod_name, "Issue": "Modifier group has 0 options."})

    # Check for duplicates
    for name in set(cat_names):
        if cat_names.count(name) > 1:
            warnings.append({"Type": "Duplicate", "Name": name, "Issue": "This category name is used more than once."})
    
    for name in set(mod_names):
        if mod_names.count(name) > 1:
            warnings.append({"Type": "Duplicate", "Name": name, "Issue": "This modifier group name is used more than once."})
    
    return warnings
# --- END NEW QA FUNCTION ---

# ============================== Streamlit UI (minimal) ==============================

tab1, tab2 = st.tabs(["PDF/Image → JSON", "Transform existing JSON"])

if "last_pdf_text" not in st.session_state:
    st.session_state.last_pdf_text = []

with tab1:
    f = st.file_uploader("Upload menu (PNG, JPG, JPEG, or PDF)", type=["png", "jpg", "jpeg", "pdf"])
    menu_name = st.text_input("Menu name", value="Generated Menu")
    price_band_id = st.text_input("Flipdish Price Band ID (required)", value="")
    attach_images = st.checkbox("Attach cropped item images (PDF only)", value=True)
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)

    if st.button("Extract and Build JSON"):
        if not price_band_id.strip():
            st.error("Price Band ID is required."); st.stop()

        loaded = load_file(f)
        if not loaded.images:
            st.error("Please upload a valid image or PDF."); st.stop()

        extracted_pages, per_page_text = [], []
        with st.spinner("Extracting..."):
            for i, im in enumerate(loaded.images):
                page_text = ""
                if loaded.is_pdf and loaded.doc is not None and fitz is not None:
                    try: page_text = loaded.doc[i].get_text("text") or ""
                    except Exception: page_text = ""
                fewshot = build_fewshot_context(page_text or menu_name)
                extracted = run_openai_single(im, model=model, fewshot=fewshot)
                extracted_pages.append(extracted)
                per_page_text.append(page_text)

        result = to_flipdish_json(
            extracted_pages,
            menu_name,
            price_band_id.strip(),
            attach_images and loaded.is_pdf, 
            loaded.doc if (loaded.is_pdf and fitz is not None) else None,
            rules=None  # auto-load rules.json silently
        )

        # Silent learning: save a few exemplars
        try:
            src_text = (per_page_text[0] if per_page_text else menu_name) or ""
            first_page = extracted_pages[0] if extracted_pages else {"categories": []}
            saved = 0
            for cat in (first_page.get("categories") or [])[:2]:
                for it in (cat.get("items") or [])[:3]:
                    save_example(src_text, {"category": cat.get("caption"), "item": it}, tags=["auto"])
                    saved += 1
        except Exception:
            pass

        # --- NEW QA CHECK ---
        st.subheader("QA Check & Warnings")
        warnings = run_qa_checks(result)
        if warnings:
            st.warning("Issues found. Please review before exporting.")
            # Display as a table
            st.dataframe(warnings, use_container_width=True)
        else:
            st.success("QA complete. No issues found!")
        # --- END QA CHECK ---

        st.subheader("Generated JSON Preview")
        st.json(result, expanded=False)
        
        fn_slug = menu_name.strip().lower()
        if not fn_slug: fn_slug = "flipdish_menu"
        fn_slug = re.sub(r'\s+', '_', fn_slug) 
        fn_slug = re.sub(r'[^a-z0-9_]', '', fn_slug) # Switched back to only underscore
        fn_slug = fn_slug or "flipdish_menu" 
        
        st.download_button(
            "Download Flipdish JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
            file_name=f"{fn_slug}.json", 
            mime="application/json"
        )

with tab2:
    jf = st.file_uploader("Upload existing JSON to re-shape", type=["json"], key="json_in")
    menu_name2 = st.text_input("Override name", key="mn2")
    price_band_id2 = st.text_input("Price Band ID", key="pb2")

    if st.button("Transform", key="btn2"):
        if not jf:
            st.error("Upload a JSON file first."); st.stop()
        if not price_band_id2.strip():
            st.error("Price Band ID is required."); st.stop()

        raw = json.load(io.BytesIO(jf.read()))
        result = to_flipdish_json([raw], menu_name2 or "", price_band_id2.strip(), False, None, rules=None)
        
        # --- NEW QA CHECK ---
        st.subheader("QA Check & Warnings")
        warnings = run_qa_checks(result)
        if warnings:
            st.warning("Issues found. Please review before exporting.")
            # Display as a table
            st.dataframe(warnings, use_container_width=True)
        else:
            st.success("QA complete. No issues found!")
        # --- END QA CHECK ---
        
        st.subheader("Generated JSON Preview")
        st.json(result, expanded=False)
        
        fn_slug_2 = (menu_name2 or "flipdish_menu").strip().lower()
        if not fn_slug_2: fn_slug_2 = "flipdish_menu"
        fn_slug_2 = re.sub(r'\s+', '_', fn_slug_2) 
        fn_slug_2 = re.sub(r'[^a-z0-9_]', '', fn_slug_2) # Switched back to only underscore
        fn_slug_2 = fn_slug_2 or "flipdish_menu" 
        
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
            file_name=f"{fn_slug_2}.json", 
            mime="application/json"
        )
