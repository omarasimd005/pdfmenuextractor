# app.py — Flipdish Menu Builder (smart title-case + intelligent modifiers + category colors + CONDITIONAL MODIFIERS)
# + Improvements:
#   - Diagnostics sidebar (env + versions + key presence)
#   - PDF raster at 300 DPI for better extraction fidelity
#   - Cached extraction per image (Streamlit cache)
#   - Learning loop: save examples, retrieve similar examples (few-shot) to guide the model
#   - Normalization rules (optional rules.json) to enforce consistent captions/min/max/aliases
#   - Clear warning when OPENAI_API_KEY is missing (prevents silent stub use)

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

import platform
import sys
import pkg_resources

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

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------- Utilities ----------------
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

def _pkg_ver(name, default="(not installed)"):
    try:
        return pkg_resources.get_distribution(name).version
    except Exception:
        return default

# ---------------- Robust loader ----------------
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
        st.error("Uploaded file is empty.")
        return LoadedFile([], None, False)

    name = (getattr(file, "name", "") or "").lower()
    if _is_pdf(data) or name.endswith(".pdf"):
        if fitz is None:
            st.error("PDF support is unavailable on this server. Upload images (PNG/JPG) or deploy with PyMuPDF.")
            return LoadedFile([], None, True)
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as e:
            st.error(f"Could not read PDF: {e}")
            return LoadedFile([], None, True)
        pages = []
        try:
            for i in range(len(doc)):
                # ↑ bump to 300 DPI for better OCR/vision fidelity
                pix = doc[i].get_pixmap(dpi=300)
                pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        except Exception as e:
            st.error(f"Could not rasterize PDF pages: {e}")
            return LoadedFile([], doc, True)
        return LoadedFile(pages, doc, True)

    try:
        img = _open_image_from_bytes(data)
        return LoadedFile([img], None, False)
    except Exception as e:
        st.error(f"Could not read image: {e}")
        return LoadedFile([], None, False)

# ---------------- Title-case helper (ignore small words) ----------------
SMALL_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in",
    "into", "nor", "of", "on", "onto", "or", "per", "the", "to", "vs",
    "via", "with", "over", "under", "up", "down", "off"
}

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

# ---------------- Category color mapping ----------------
CATEGORY_COLOR_RULES = [
    (re.compile(r"^(starter|starters)$", re.I), "#E67E22"),
    (re.compile(r"^(main|mains)$", re.I), "#C0392B"),
    (re.compile(r"^(side|sides)$", re.I), "#FBC02D"),
    (re.compile(r"^(soup|soups)\s*/\s*(salad|salads)$", re.I), "#2E8B57"),
    (re.compile(r"^(soup|soups)$", re.I), "#2E8B57"),
    (re.compile(r"^(salad|salads)$", re.I), "#2E8B57"),
    (re.compile(r"^(dessert|desserts)$", re.I), "#8E44AD"),
    (re.compile(r"^(beverage|beverages|drink|drinks)$", re.I), "#9b9b9b"),
    (re.compile(r"^(special|specials)$", re.I), "#FF66B2"),
    (re.compile(r"^(kid|kids)\s*(menu)?$", re.I), "#3498DB"),
    (re.compile(r"^(pizza|pizzas)$", re.I), "#D64541"),
    (re.compile(r"^(burger|burgers)$", re.I), "#935116"),
    (re.compile(r"^(sauce|sauces)$", re.I), "#FFDAB9"),
    (re.compile(r"^(wine|wines|spirit|spirits|wines\s*/\s*spirits)$", re.I), "#2C3E50"),
]

def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _relative_luminance(rgb: Tuple[int, int, int]) -> float:
    def _linearize(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (_linearize(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def pick_category_colors(caption: str) -> Optional[Dict[str, str]]:
    if not caption:
        return None
    norm = caption.strip()
    for pat, bg in CATEGORY_COLOR_RULES:
        if pat.match(norm):
            lum = _relative_luminance(_hex_to_rgb(bg))
            fg = "#FFFFFF" if lum < 0.5 else "#000000"
            return {"backgroundColor": bg, "foregroundColor": fg}
    joined = re.sub(r"\s+", " ", norm, flags=re.I)
    for pat, bg in CATEGORY_COLOR_RULES:
        if pat.match(joined):
            lum = _relative_luminance(_hex_to_rgb(bg))
            fg = "#FFFFFF" if lum < 0.5 else "#000000"
            return {"backgroundColor": bg, "foregroundColor": fg}
    return None

# ---------------- Parsing helpers ----------------
PRICE_RE = re.compile(r'(?:£|\$|€)?\s*(\d{1,3}(?:\.\d{1,2})?)')
PLUS_PRICE_RE = re.compile(r'(?i)^(?P<name>.*?\S)\s*(?:\+|plus\s*)(?P<price>\d+(?:\.\d+)?)\s*$')
PENCE_RE = re.compile(r'(\d{1,3})\s*(?:p|P)\b')

def parse_price_from_text(*texts: str) -> Optional[float]:
    for t in texts or []:
        if not t:
            continue
        m = PRICE_RE.search(t)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
        if PENCE_RE.search(t):
            continue
    return None

_SPLIT_PATTERNS = [
    re.compile(r"^(?P<name>.+?)\s*[-–—:]\s*(?P<desc>.+)$"),
    re.compile(r"^(?P<name>.+?)\s*\((?P<desc>[^)]+)\)\s*$"),
]
def split_caption_and_inline_notes(text: str) -> Tuple[str, str]:
    t = (text or "").strip()
    for p in _SPLIT_PATTERNS:
        m = p.match(t)
        if m:
            return m.group("name").strip(), m.group("desc").strip()
    return t, ""

# ---------------- “Learning” store (examples + retrieval) ----------------
EXAMPLES_PATH = "examples.jsonl"  # appended JSON lines
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
    with open(EXAMPLES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_examples() -> List[dict]:
    if not os.path.exists(EXAMPLES_PATH):
        return []
    out = []
    with open(EXAMPLES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
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
    if not q:
        return []
    exs = load_examples()
    scored = []
    for ex in exs:
        s = _cos(q, _bow(ex.get("source","")))
        if s > 0:
            scored.append((s, ex))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [ex for _, ex in scored[:k]]

def build_fewshot_context(query_text: str) -> str:
    shots = top_k_examples(query_text, k=3)
    if not shots:
        return ""
    lines = []
    for ex in shots:
        lines.append(json.dumps({
            "source_excerpt": (ex.get("source","") or "")[:400],
            "expected_flipdish_piece": ex.get("flipdish",{})
        }, ensure_ascii=False))
    return "\n".join(lines)

# ---------------- Vision extraction (STRICT JSON with modifiers + CONDITIONALS) ----------------
BASE_EXTRACTION_PROMPT = """
You output ONLY JSON (no markdown) with this schema:

{
  "name": string,
  "categories": [
    {
      "caption": string,
      "items": [
        {
          "caption": string,
          "description": string,
          "notes": string,
          "price": number,
          "modifiers": [
            {
              "caption": string,
              "min": number|null,
              "max": number|null,
              "options": [
                {
                  "caption": string,
                  "price": number|null,
                  "modifiers": [
                    {
                      "caption": string,
                      "min": number|null,
                      "max": number|null,
                      "options": [{"caption": string, "price": number|null}]
                    }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

Detect modifiers from many phrasings:
- "GOES WELL WITH X +Y, Z +W"
- "Add ham +3", "Add chicken or chorizo +4", "Add steak +6"
- "Grilled chicken +8.5", "Smoked salmon +7.5" (treat as add-ons)
- "Choice of toast or pancakes", "Choose from ...", "Comes with choice of ..."
- "Served with ... (+X to upgrade ...)" (treat upgrades as options with price)
- CONDITIONAL examples: "If you choose X, pick a sauce", "Select size -> then choose sides". In these cases, attach the follow-up groups under the chosen option's "modifiers".

Rules:
- Item price must be numeric; ignore currency symbols.
- Options without explicit price -> price=null (0).
- Keep headings that have a price as items; ignore ALL-CAPS section headers with no price.
"""

def _img_hash(img: Image.Image) -> str:
    return hashlib.blake2b(img.tobytes(), digest_size=16).hexdigest()

@st.cache_data(show_spinner=False, ttl=7*24*3600)
def _cached_extract_page(img_bytes: bytes, model: str, fewshot: str) -> Dict[str, Any]:
    # Recreate PIL image inside the cache function
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return _run_openai_single_uncached(im, model=model, fewshot=fewshot)

def run_openai_single(img: Image.Image, model: str = "gpt-4o", fewshot: str = "") -> Dict[str, Any]:
    # Wrapper to use cache
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return _cached_extract_page(buf.getvalue(), model, fewshot)

def _run_openai_single_uncached(image: Image.Image, model: str = "gpt-4o", fewshot: str = "") -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        # Offline stub (so pipeline runs without API key)
        return {
            "name": "Sample",
            "categories": [{
                "caption": "brunch",
                "items": [
                    {
                        "caption": "chilaquiles",
                        "description": "Add chicken or chorizo +4 / Add steak +6",
                        "price": 13,
                        "modifiers": [
                            {"caption": "ADD", "min": 0, "max": 3, "options": [
                                {"caption": "chicken", "price": 4},
                                {"caption": "chorizo", "price": 4},
                                {"caption": "steak", "price": 6}
                            ]}
                        ]
                    },
                    {
                        "caption": "Combo Breakfast",
                        "description": "Choose your main; if Waffles, pick a syrup",
                        "price": 28,
                        "modifiers": [
                            {
                                "caption": "Choose Main",
                                "min": 1,
                                "max": 1,
                                "options": [
                                    {"caption": "Pancakes", "price": None},
                                    {"caption": "Waffles", "price": None, "modifiers": [
                                        {
                                            "caption": "Choose Syrup",
                                            "min": 1,
                                            "max": 1,
                                            "options": [{"caption": "Maple", "price": None}, {"caption": "Chocolate", "price": None}]
                                        }
                                    ]}
                                ]
                            }
                        ]
                    }
                ]
            }]
        }

    # Build the messages with optional few-shot context
    sys_prompt = BASE_EXTRACTION_PROMPT
    if fewshot:
        sys_prompt = sys_prompt + "\n\nEXAMPLES (follow this structure and style):\n" + fewshot

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

# ---------------- PDF helpers (find image near item name) ----------------
def find_item_rects(page: "fitz.Page", name_clean: str) -> List["fitz.Rect"]:
    if fitz is None or not name_clean.strip():
        return []
    r = page.search_for(name_clean)
    if r:
        return r
    toks = name_clean.split()
    for n in (3, 2, 1):
        if len(toks) >= n:
            r = page.search_for(" ".join(toks[:n]))
            if r:
                return r
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
    if not best:
        return None
    clip = fitz.Rect(best.x0 - margin, best.y0 - margin, best.x1 + margin, best.y1 + margin)
    pix = page.get_pixmap(clip=clip, dpi=300)
    return pix.tobytes("png")

# ---------------- Regex fallback for missed modifiers ----------------
MODIFIER_HEADERS = [
    r"goes\s+well\s+with", r"goes\s+with", r"add[-\s]*ons?", r"addons?", r"extras?",
]
MODIFIER_HEADER_RE = re.compile(r"(?i)\b(" + "|".join(MODIFIER_HEADERS) + r")\b[:\s]*")
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
            if t.isupper() and len(t.split()) <= 5 and not PRICE_RE.search(t):
                break
            pm = PLUS_PRICE_RE.match(t)
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
        pm = PLUS_PRICE_RE.match(line.strip())
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
        seen = set()
        opts = []
        for n, p in items:
            key = (n.lower(), p if p is not None else -1)
            if key in seen:
                continue
            seen.add(key)
            opts.append({"caption": n, "price": p})
        if opts:
            out.append({"caption": cap, "min": None, "max": None, "options": opts})
    return out

# ---------------- Flipdish JSON builder (supports CONDITIONAL MODIFIERS) + normalization ----------------
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
        key = key_raw.strip().upper()
        nowh = now_iso_hms()
        if key in modifiers_index:
            return modifiers_index[key]
        g = {
            "etag": f"W/\"datetime'{nowz}'\"",
            "timestamp": nowh,
            "canSameItemBeSelectedMultipleTimes": True if can_repeat is None else bool(can_repeat),
            "caption": smart_title(key_raw if key_raw else "ADD"),
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
        if not grp:
            return
        g_caption = smart_title(grp.get("caption") or "ADD")
        g_min = grp.get("min")
        g_max = grp.get("max")
        can_repeat = grp.get("canSameItemBeSelectedMultipleTimes")
        group = ensure_group(g_caption, g_min, g_max, can_repeat)

        for opt in (grp.get("options") or []):
            oname = smart_title((opt.get("caption") or "").strip())
            if not oname:
                continue
            price = opt.get("price")
            opt_item = _ensure_option_item(group, oname, price)
            for child_grp in (opt.get("modifiers") or []):
                _process_group(opt_item, child_grp)

        _link_group_to_parent(parent_entity, group)

    cat_index: Dict[str, Any] = {}

    for page_i, data in enumerate(extracted_pages):
        for cat_in in (data.get("categories") or []):
            cat_caption_raw = (cat_in.get("caption") or "Category").strip()
            cat_caption = smart_title(cat_caption_raw).upper()
            ck = cat_caption.lower()
            if ck not in cat_index:
                cat = {
                    "etag": f"W/\"datetime'{nowz}'\"",
                    "timestamp": now_iso_hms(),
                    "caption": cat_caption,
                    "enabled": True,
                    "id": guid(),
                    "items": [],
                    "overrides": []
                }
                colors = pick_category_colors(cat_caption_raw)
                if colors:
                    cat["backgroundColor"] = colors["backgroundColor"]
                    cat["foregroundColor"] = colors["foregroundColor"]

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

    # Normalization rules
    if rules:
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
                return target
        return n or "ADD"

    for g in flipdish_json.get("modifiers", []):
        g["caption"] = canon_mod_name(g.get("caption"))
        if g["caption"] in force:
            mm = force[g["caption"]]
            if "min" in mm: g["min"] = int(mm["min"])
            if "max" in mm: g["max"] = int(mm["max"])
        for it in g.get("items", []):
            label = (it.get("caption","") or "").strip().lower()
            for canon, alist in opt_alias.items():
                if label in [canon] + alist:
                    it["caption"] = canon.title()
    return flipdish_json

# ---------------- Streamlit UI ----------------
load_dotenv()
st.set_page_config(page_title="Flipdish Menu Builder", page_icon="🍽️", layout="centered")
st.title("Flipdish Menu Builder — smarter extraction, conditional modifiers, and learning")

# Diagnostics
with st.sidebar:
    st.subheader("Diagnostics")
    st.write({
        "env": os.environ.get("STREAMLIT_RUNTIME", "local"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "OPENAI_API_KEY set?": bool(os.getenv("OPENAI_API_KEY")),
        "streamlit": _pkg_ver("streamlit"),
        "openai": _pkg_ver("openai"),
        "Pillow": _pkg_ver("Pillow"),
        "PyMuPDF": _pkg_ver("PyMuPDF"),
    })
    st.caption("Examples & rules are stored in app root (examples.jsonl, rules.json).")

tab1, tab2 = st.tabs(["PDF/Image → JSON", "Transform existing JSON"])

# Keep small global state for learning
if "last_extracted" not in st.session_state:
    st.session_state.last_extracted = None   # list of per-page LLM outputs
if "last_flipdish" not in st.session_state:
    st.session_state.last_flipdish = None
if "last_pdf_text" not in st.session_state:
    st.session_state.last_pdf_text = []      # per-page text (for PDFs)

with tab1:
    st.subheader("1) Upload your menu (PDF recommended for image extraction)")
    f = st.file_uploader("Upload image or PDF", type=["png", "jpg", "jpeg", "pdf"])
    st.subheader("2) Options")
    menu_name = st.text_input("Menu name", value="Generated Menu")
    price_band_id = st.text_input("Flipdish Price Band ID (required)", value="")
    attach_images = st.checkbox("Attach cropped item images (PDF only)", value=True)
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)

    # Optional rules.json uploader
    st.markdown("**(Optional)**: Upload normalization `rules.json`")
    rules_file = st.file_uploader("rules.json", type=["json"], key="rules")
    rules_obj = DEFAULT_RULES
    if rules_file:
        try:
            rules_obj = json.load(io.BytesIO(rules_file.read()))
            st.success("Loaded rules.json")
        except Exception as e:
            st.warning(f"Could not parse rules.json: {e}. Using defaults.")

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. The app will use an offline stub and produce minimal output.")

    if st.button("Extract and Build JSON"):
        if not price_band_id.strip():
            st.error("Price Band ID is required.")
            st.stop()
        loaded = load_file(f)
        if not loaded.images:
            st.error("No valid pages/images found.")
            st.stop()

        extracted_pages = []
        per_page_text = []

        with st.spinner("Extracting with Vision…"):
            # Build few-shot context text using PDF page text (preferred) or menu_name
            for i, im in enumerate(loaded.images):
                page_text = ""
                if loaded.is_pdf and loaded.doc is not None and fitz is not None:
                    try:
                        page_text = loaded.doc[i].get_text("text") or ""
                    except Exception:
                        page_text = ""
                query_for_examples = page_text or menu_name
                fewshot = build_fewshot_context(query_for_examples)

                # Cached call
                extracted = run_openai_single(im, model=model, fewshot=fewshot)
                extracted_pages.append(extracted)
                per_page_text.append(page_text)

        result = to_flipdish_json(
            extracted_pages,
            menu_name,
            price_band_id.strip(),
            attach_images and loaded.is_pdf,
            loaded.doc if (loaded.is_pdf and fitz is not None) else None,
            rules=rules_obj
        )

        st.session_state.last_extracted = extracted_pages
        st.session_state.last_flipdish = result
        st.session_state.last_pdf_text = per_page_text

        st.success("✅ Flipdish JSON created")
        st.json(result, expanded=False)
        st.download_button(
            "Download Flipdish JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
            file_name="flipdish_menu.json",
            mime="application/json"
        )
        st.download_button(
            "Download Extracted (pre-Flipdish) JSON",
            data=json.dumps(extracted_pages, indent=2, ensure_ascii=False).encode(),
            file_name="extracted_raw.json",
            mime="application/json"
        )

        # Save examples (learning)
        with st.expander("Save examples to improve future runs"):
            tags = st.text_input("Tags (comma separated)", value="default")
            if st.button("Save a few examples (top of first page)"):
                try:
                    src_text = (per_page_text[0] if per_page_text else menu_name) or ""
                    first_page = extracted_pages[0] if extracted_pages else {"categories": []}
                    # Save up to 3 items from the first category as exemplars
                    saved = 0
                    for cat in (first_page.get("categories") or [])[:2]:
                        for it in (cat.get("items") or [])[:3]:
                            save_example(src_text, {"category": cat.get("caption"), "item": it}, [t.strip() for t in tags.split(",") if t.strip()])
                            saved += 1
                    st.success(f"Saved {saved} example(s) to {EXAMPLES_PATH}. Future extractions will use them as few-shot guidance.")
                except Exception as e:
                    st.error(f"Could not save examples: {e}")

with tab2:
    st.subheader("Re-shape existing JSON (no image extraction)")
    jf = st.file_uploader("Upload existing JSON", type=["json"], key="json_in")
    menu_name2 = st.text_input("Override name", key="mn2")
    price_band_id2 = st.text_input("Price Band ID", key="pb2")

    if st.button("Transform", key="btn2"):
        if not jf:
            st.error("Upload a JSON file first.")
            st.stop()
        if not price_band_id2.strip():
            st.error("Price Band ID is required.")
            st.stop()
        raw = json.load(io.BytesIO(jf.read()))
        result = to_flipdish_json([raw], menu_name2 or "", price_band_id2.strip(), False, None, rules=DEFAULT_RULES)
        st.success("✅ Re-shaped successfully")
        st.json(result, expanded=False)
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
            file_name="flipdish_menu.json",
            mime="application/json"
        )
