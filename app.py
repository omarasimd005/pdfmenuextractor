# app.py — Flipdish Menu Builder (smart title-case + intelligent modifiers + category colors + CONDITIONAL MODIFIERS)
# - Robust loader (PDF/JPG/JPEG/PNG)
# - Vision extraction (strict JSON) with explicit per-item modifiers
# - NEW: Conditional modifiers (options can include nested "modifiers" → become option-level modifierMembers)
# - Regex fallback for missed modifiers (Add/Choice/Goes well with/etc.)
# - Prices + descriptions (stored in `notes`)
# - Base64 item images from PDF (nearest crop) in imageUrl
# - Flipdish schema with global modifiers + modifierMembers
# - Smart title-case for item/category captions (skips small words unless first/last)
# - Auto category colors (backgroundColor + foregroundColor) based on caption

import base64
import io
import json
import os
import uuid
import datetime
import re
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from PIL import Image, ImageFile
import fitz  # PyMuPDF
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

# ---------------- Robust loader ----------------
def _is_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF")

def _open_image_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im.load()
    return im.convert("RGB")

class LoadedFile:
    def __init__(self, images: List[Image.Image], doc: Optional[fitz.Document], is_pdf: bool):
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

    name = (file.name or "").lower()
    if _is_pdf(data) or name.endswith(".pdf"):
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as e:
            st.error(f"Could not read PDF: {e}")
            return LoadedFile([], None, True)
        pages = []
        try:
            for i in range(len(doc)):
                pix = doc[i].get_pixmap(dpi=200)
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
    # Title-case each hyphenated part: "fresh-baked" -> "Fresh-Baked"
    return "-".join(p.capitalize() if p else p for p in token.split("-"))

def smart_title(text: str) -> str:
    """
    Title-case while skipping small words unless first/last.
    Keeps existing internal capitalization for ALLCAPS acronyms.
    """
    if not text:
        return text
    tokens = re.split(r'(\s+)', text.strip())
    # words-only length for first/last detection
    words_only = [t for t in tokens if not re.match(r'\s+', t)]
    result = []
    word_index = 0
    for t in tokens:
        if re.match(r'\s+', t):
            result.append(t)
            continue
        lower = t.lower()
        # Preserve ALLCAPS (e.g., BBQ, GF)
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
    """
    Returns {"backgroundColor": "#RRGGBB", "foregroundColor": "#FFFFFF|#000000"} or None.
    """
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

# ---------------- Vision extraction (STRICT JSON with modifiers + CONDITIONALS) ----------------
EXTRACTION_PROMPT = """
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
                  "modifiers": [  // OPTIONAL: conditional groups that appear when THIS option is selected
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

def run_openai_single(image: Image.Image, model: str = "gpt-4o") -> Dict[str, Any]:
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

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract menu JSON with explicit and conditional modifiers for this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(image)}"}}
            ]}
        ],
    )
    return json.loads(resp.choices[0].message.content)

# ---------------- PDF image crop (nearest to item name) ----------------
def find_item_rects(page: fitz.Page, name_clean: str) -> List[fitz.Rect]:
    if not name_clean.strip():
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

def nearest_image_crop(page: fitz.Page, near: fitz.Rect, margin: float = 12.0) -> Optional[bytes]:
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
    pix = page.get_pixmap(clip=clip, dpi=200)
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

    # Header groups (GOES WELL WITH / EXTRAS / ADD-ONS ...)
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

    # "Add chicken or chorizo +4"
    for m in ADD_PATTERN.finditer(text):
        names = split_option_list(m.group("opts"))
        price = float(m.group("price")) if m.group("price") else None
        if names:
            groups.setdefault("ADD", [])
            for n in names:
                groups["ADD"].append((n, price))

    # Standalone tokens: "Grilled chicken +8.5"
    for line in re.split(r"[.;\n]+", text):
        pm = PLUS_PRICE_RE.match(line.strip())
        if pm:
            groups.setdefault("ADD", []).append((pm.group("name").strip(" -–—:"), float(pm.group("price"))))

    # Choice of ...
    for m in CHOICE_PATTERN.finditer(text):
        names = split_option_list(m.group("opts"))
        if names:
            groups.setdefault("CHOICE OF", [])
            for n in names:
                groups["CHOICE OF"].append((n, 0.0))

    # Dedupe + convert to LLM-like structure
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

# ---------------- Flipdish JSON builder (supports CONDITIONAL MODIFIERS) ----------------
def to_flipdish_json(
    extracted_pages: List[Dict[str, Any]],
    menu_name: str,
    price_band_id: str,
    attach_pdf_images: bool,
    src_pdf_doc: Optional[fitz.Document]
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

    # global modifier groups registry
    modifiers_index: Dict[str, Dict[str, Any]] = {}

    def ensure_group(caption: str, min_sel: Optional[int] = None, max_sel: Optional[int] = None, can_repeat: Optional[bool] = None) -> Dict[str, Any]:
        """
        Ensure a global modifier group exists, keyed by UPPER caption.
        Allow min/max/canSameItemBeSelectedMultipleTimes to be set the first time.
        """
        key_raw = caption or "ADD"
        key = key_raw.strip().upper()
        nowh = now_iso_hms()
        if key in modifiers_index:
            g = modifiers_index[key]
            # if min/max provided later, keep first-set values (Flipdish often expects consistency)
            return g
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
        g["_items_map"] = {}  # name(lower)|priceKey -> opt
        modifiers_index[key] = g
        return g

    # -- recursion helpers for conditional modifiers -----------------
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
            "modifierMembers": [],  # critical for CONDITIONALS
            "overrides": []
        }
        group["_items_map"][key_nm] = opt_item
        return opt_item

    def _link_group_to_parent(parent_entity: Dict[str, Any], group_obj: Dict[str, Any]) -> None:
        # parent_entity is either an ITEM or an OPTION ITEM; attach a modifierMember referencing the global group
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
        """
        Create/ensure a global modifier group, add all options, link it to the parent entity,
        then process any OPTION-level nested 'modifiers' recursively.
        """
        if not grp:
            return
        g_caption = smart_title(grp.get("caption") or "ADD")
        g_min = grp.get("min")
        g_max = grp.get("max")
        can_repeat = grp.get("canSameItemBeSelectedMultipleTimes")
        group = ensure_group(g_caption, g_min, g_max, can_repeat)

        # Add options
        for opt in (grp.get("options") or []):
            oname = smart_title((opt.get("caption") or "").strip())
            if not oname:
                continue
            price = opt.get("price")
            opt_item = _ensure_option_item(group, oname, price)

            # CONDITIONAL: nested modifiers under this option
            child_mods = opt.get("modifiers") or []
            for child_grp in child_mods:
                _process_group(opt_item, child_grp)

        # Finally, link the group to the parent (item or option)
        _link_group_to_parent(parent_entity, group)

    # ---------------------------------------------------------------

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
                if page is not None and name:
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

                # Use LLM-provided modifiers first; fallback to regex if empty
                llm_mods = it.get("modifiers") or []
                if not llm_mods:
                    llm_mods = fallback_extract_modifiers("\n".join([desc, notes, inline]))

                # Process each group (handles conditional recursion)
                for grp in llm_mods:
                    _process_group(item, grp)

                cat["items"].append(item)

    # finalize modifiers
    for g in modifiers_index.values():
        g["items"] = list(g["_items_map"].values())
        del g["_items_map"]
        out["modifiers"].append(g)

    out["categories"] = [c for c in out["categories"] if c.get("items")]
    return out

# ---------------- Streamlit UI ----------------
load_dotenv()
st.set_page_config(page_title="Flipdish Menu Builder", page_icon="🍽️", layout="centered")
st.title("Flipdish Menu Builder — smart title-case + modifiers + category colors")
st.caption("Extracts items, prices, descriptions, base64 images, intelligent modifiers (incl. conditional), and auto category colors.")

tab1, tab2 = st.tabs(["PDF/Image → JSON", "Transform existing JSON"])

with tab1:
    st.subheader("1) Upload your menu (PDF recommended for image extraction)")
    f = st.file_uploader("Upload image or PDF", type=["png", "jpg", "jpeg", "pdf"])
    st.subheader("2) Options")
    menu_name = st.text_input("Menu name", value="Generated Menu")
    price_band_id = st.text_input("Flipdish Price Band ID (required)", value="")
    attach_images = st.checkbox("Attach cropped item images (PDF only)", value=True)
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)

    if st.button("Extract and Build JSON"):
        if not price_band_id.strip():
            st.error("Price Band ID is required.")
            st.stop()
        loaded = load_file(f)
        if not loaded.images:
            st.error("No valid pages/images found.")
            st.stop()

        extracted_pages = []
        with st.spinner("Extracting with Vision…"):
            for im in loaded.images:
                extracted_pages.append(run_openai_single(im, model=model))

        result = to_flipdish_json(
            extracted_pages,
            menu_name,
            price_band_id.strip(),
            attach_images and loaded.is_pdf,
            loaded.doc if loaded.is_pdf else None
        )

        st.success("✅ Flipdish JSON created")
        st.json(result, expanded=False)
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
            file_name="flipdish_menu.json",
            mime="application/json"
        )

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
        # Note: We let to_flipdish_json handle structure; if the uploaded JSON is already in our
        # extractor's shape, it will be converted. If it's a Flipdish-like shape, ensure fields align.
        result = to_flipdish_json([raw], menu_name2 or "", price_band_id2.strip(), False, None)
        st.success("✅ Re-shaped successfully")
        st.json(result, expanded=False)
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
            file_name="flipdish_menu.json",
            mime="application/json"
        )
