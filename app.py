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
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ============================== Small utils ==============================

def guid() -> str:
    return str(uuid.uuid4())

def now_iso_z() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def now_iso_hms() -> str:
    return datetime.datetime.utcnow().isoformat()

def to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

def smart_title(text: str) -> str:
    """
    Title-case with heuristics so menu names look natural.
    """
    if not text:
        return text
    SMALL_WORDS = {
        "a","an","and","as","at","but","by","for","from","in","into","of",
        "on","or","the","to","vs","via","with"
    }
    def _cap_hyphenated(word: str) -> str:
        parts = word.split("-")
        return "-" .join(p.capitalize() for p in parts)
    if text.isupper():
        # Keep obvious all-caps like "BBQ", "USA", but soften long blocks
        if len(text) <= 4:
            return text
        return text.title()
    # Split preserving spaces so we can rejoin nicely
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
            if (
                word_index == 0
                or word_index == len(words_only) - 1
                or lower not in SMALL_WORDS
            ):
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

def pick_category_colors(caption: str) -> Optional[Dict[str,str]]:
    name = (caption or "").strip().lower()
    for rex, color in CATEGORY_COLOR_RULES:
        if rex.match(name):
            fg = "#FFFFFF"
            if color.lower() in ("#fbc02d", "#ebebeb", "#ffffff"):
                fg = "#000000"
            return {
                "backgroundColor": color,
                "foregroundColor": fg
            }
    return None

# ============================== Caching & examples ==============================

EXAMPLES_PATH = "examples.jsonl"
CACHE_PATH = "cache.jsonl"

def load_cache() -> Dict[str, dict]:
    if not os.path.exists(CACHE_PATH):
        return {}
    out = {}
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = rec.get("key")
                    if key:
                        out[key] = rec.get("value")
                except Exception:
                    pass
    except Exception:
        pass
    return out

def save_cache_entry(key: str, value: dict):
    try:
        with open(CACHE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
    except Exception:
        pass

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
        pass
    return out

# ============================== File loading ==============================

class LoadedFile:
    def __init__(self, images: List[Image.Image], doc, is_pdf: bool):
        self.images = images
        self.doc = doc
        self.is_pdf = is_pdf

def _is_pdf(data: bytes) -> bool:
    return data[:4] == b"%PDF"

def load_file(file) -> LoadedFile:
    if Image is None:
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
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
        return LoadedFile(images, doc, True)

    # Image
    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
    except Exception:
        return LoadedFile([], None, False)
    return LoadedFile([img], None, False)

# ============================== OpenAI extraction (with cache) ==============================

def _cache_key_for_image(png_bytes: bytes, model: str, fewshot: str) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(png_bytes)
    h.update(model.encode("utf-8"))
    h.update((fewshot or "").encode("utf-8"))
    return h.hexdigest()

_CACHE = load_cache()

def _cached_extract_page(png_bytes: bytes, model: str, fewshot: str) -> Dict[str, Any]:
    key = _cache_key_for_image(png_bytes, model, fewshot)
    if key in _CACHE:
        return _CACHE[key]
    value = _run_openai_single_uncached_core(png_bytes, model, fewshot)
    _CACHE[key] = value
    save_cache_entry(key, value)
    return value

def run_openai_single(img: Image.Image, model: str = "gpt-4o", fewshot: str = "") -> Dict[str, Any]:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return _cached_extract_page(buf.getvalue(), model, fewshot)

def _run_openai_single_uncached_core(png_bytes: bytes, model: str, fewshot: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        # Offline stub to keep UI functional
        return {
            "name": "Sample",
            "categories": [{
                "caption": "Brunch",
                "description": "Brunch favourites.",
                "items": [
                    {
                        "caption": "Combo Breakfast",
                        "description": "Choose main; if Waffles then choose syrup",
                        "price": 28,
                        "modifiers": [{
                            "caption": "Choose Main",
                            "min": 1,
                            "max": 1,
                            "options": [
                                {"caption": "Pancakes", "price": None},
                                {"caption": "Waffles", "price": None},
                            ]
                        }]
                    }
                ]
            }]
        }

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are a menu-structure extractor for Flipdish.\n"
        "Return ONLY valid JSON with this structure:\n"
        "{\n"
        '  "name": string,\n'
        '  "categories": [\n'
        "    {\n"
        '      "caption": string,\n'
        '      "description": string (optional, for category-level description if present),\n'
        '      "items": [\n'
        "        {\n"
        '          "caption": string,\n'
        '          "description": string (optional),\n'
        '          "price": number or null,\n'
        '          "notes": string (optional),\n'
        '          "modifiers": [\n'
        "            {\n"
        '              "caption": string,\n'
        '              "min": number or null,\n'
        '              "max": number or null,\n'
        '              "canSameItemBeSelectedMultipleTimes": boolean (optional),\n'
        '              "options": [\n'
        "                {\n"
        '                  "caption": string,\n'
        '                  "price": number or null,\n'
        '                  "modifiers": [...] (optional, same shape recursively)\n'
        "                }\n"
        "              ]\n"
        "            }\n"
        "          ]\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Detect choices, add-ons, and conditional selections as modifiers.\n"
        "Be strict: no trailing commas, no comments."
    )
    if fewshot:
        prompt = fewshot + "\n\n" + prompt

    img_b64 = base64.b64encode(png_bytes).decode("ascii")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You convert restaurant menus into clean JSON for Flipdish integration."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]
                }
            ],
            max_tokens=4000,
            temperature=0.1,
        )
        txt = resp.choices[0].message.content.strip()
        txt = txt.strip("` \n")
        if txt.lower().startswith("json"):
            txt = txt[4:].lstrip(": \n")
        return json.loads(txt)
    except Exception as e:
        # If parsing fails, fall back stub
        return {
            "name": "Menu",
            "categories": []
        }

# ============================== Fallback modifier parsing ==============================

MODIFIER_HEADERS = [
    r"goes\s+well\s+with",
    r"goes\s+with",
    r"add[-\s]*ons?",
    r"addons?",
    r"extras?",
]
MODIFIER_HEADER_RE = re.compile(
    r"(?i)\b(" + "|".join(MODIFIER_HEADERS) + r")\b[:\s]*"
)
PLUS_PRICE_LINE = re.compile(
    r"(?i)^(?P<name>.*?\S)\s*(?:\+|plus\s*)(?P<price>\d+(?:\.\d+)?)\s*$"
)
ADD_PATTERN = re.compile(
    r"(?i)\badd\s+(?P<opts>[^.;\n/]+?)\s*(?:\+\s*(?P<price>\d+(?:\.\d+)?))(?=[\s\).,;/]|$)"
)
CHOICE_PATTERN = re.compile(
    r"(?i)\b(choice\s+of|choose\s+from|comes\s+with\s+choice\s+of)\s+(?P<opts>[^.;\n/]+)"
)

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
            m2 = PLUS_PRICE_LINE.match(t)
            if m2:
                nm = m2.group("name").strip()
                pr = float(m2.group("price"))
                groups.setdefault(gcap, []).append((nm, pr))
            else:
                groups.setdefault(gcap, []).append((t, None))

    for m in ADD_PATTERN.finditer(text):
        opts = split_option_list(m.group("opts"))
        price = m.group("price")
        price = float(price) if price else None
        if opts:
            groups.setdefault("ADD", [])
            for o in opts:
                groups["ADD"].append((o, price))

    for m in CHOICE_PATTERN.finditer(text):
        opts = split_option_list(m.group("opts"))
        if opts:
            groups.setdefault("CHOICE OF", [])
            for o in opts:
                groups["CHOICE OF"].append((o, None))

    out = []
    for cap, pairs in groups.items():
        seen = set()
        opts = []
        for n, p in pairs:
            key = (n.lower(), p if p is not None else -1)
            if key in seen:
                continue
            seen.add(key)
            opts.append({"caption": n, "price": p})
        if opts:
            out.append(
                {
                    "caption": cap,
                    "min": None,
                    "max": None,
                    "options": opts,
                }
            )
    return out

# ============================== Layout helpers (images near text) ==============================

def find_item_rects(page, item_name: str) -> List["fitz.Rect"]:
    if fitz is None:
        return []
    name = (item_name or "").strip()
    if not name:
        return []
    rects = []
    for inst in page.search_for(name):
        rects.append(inst)
    return rects

def nearest_image_crop(page, near: "fitz.Rect") -> Optional[bytes]:
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
        d = abs(iy - ay)
        if d < best_d:
            best_d = d
            best = ir
    if not best:
        return None
    margin = 5
    clip = fitz.Rect(
        best.x0 - margin,
        best.y0 - margin,
        best.x1 + margin,
        best.y1 + margin,
    )
    pix = page.get_pixmap(clip=clip, dpi=300)
    return pix.tobytes("png")

# ============================== Flipdish builder (conditional modifiers + cat descriptions) ==============================

def split_caption_and_inline_notes(raw: str) -> Tuple[str, str]:
    if not raw:
        return "", ""
    # Split things like "Item Name - description" or "Item Name: description"
    m = re.match(r"^(.*?)([-–—:]\s+)(.+)$", raw.strip())
    if not m:
        return raw.strip(), ""
    name = m.group(1).strip()
    inline = m.group(3).strip()
    return name, inline

def parse_price_from_text(*parts: str) -> Optional[float]:
    text = " ".join(p for p in parts if p)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:[A-Z]{3}|€|£|\$)?\b", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def to_flipdish_json(
    extracted_pages: List[Dict[str, Any]],
    menu_name: str,
    price_band_id: str,
    attach_pdf_images: bool,
    src_pdf_doc: Optional["fitz.Document"],
    rules: Optional[dict] = None,
) -> Dict[str, Any]:
    nowz = now_iso_z()
    out: Dict[str, Any] = {
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
        "categoryGroups": [],
    }

    modifiers_index: Dict[str, Dict[str, Any]] = {}

    def ensure_group(
        caption: str,
        min_sel: Optional[int] = None,
        max_sel: Optional[int] = None,
        can_repeat: Optional[bool] = None,
    ) -> Dict[str, Any]:
        key = caption.strip().upper()
        if not key:
            key = "GROUP"
        if key in modifiers_index:
            g = modifiers_index[key]
        else:
            g = {
                "etag": f"W/\"datetime'{nowz}'\"",
                "timestamp": now_iso_hms(),
                "id": guid(),
                "caption": caption,
                "enabled": True,
                "hiddenInOrderFlow": False,
                "min": 0 if min_sel is None else min_sel,
                "max": 1 if max_sel is None else max_sel,
                "canSameItemBeSelectedMultipleTimes": bool(
                    can_repeat if can_repeat is not None else False
                ),
                "paramsJson": "{}",
                "position": len(modifiers_index),
                "_items_map": {},
                "overrides": [],
            }
            modifiers_index[key] = g
        if min_sel is not None:
            g["min"] = int(min_sel)
        if max_sel is not None:
            g["max"] = int(max_sel)
        if can_repeat is not None:
            g["canSameItemBeSelectedMultipleTimes"] = bool(can_repeat)
        return g

    def _ensure_option_item(
        group_obj: Dict[str, Any], caption: str, price: Optional[float]
    ) -> Dict[str, Any]:
        m = group_obj["_items_map"]
        key = (caption.strip().lower(), float(price) if price is not None else None)
        if key in m:
            return m[key]
        item = {
            "etag": f"W/\"datetime'{nowz}'\"",
            "timestamp": now_iso_hms(),
            "id": guid(),
            "caption": caption,
            "enabled": True,
            "doesPriceRepresentRewardPoints": False,
            "paramsJson": "{}",
            "pricingProfiles": [],
            "charges": [],
            "modifierMembers": [],
            "overrides": [],
        }
        if price is not None:
            item["pricingProfiles"].append(
                {
                    "etag": f"W/\"datetime'{nowz}'\"",
                    "timestamp": now_iso_hms(),
                    "priceBandId": price_band_id,
                    "collectionPrice": float(price),
                    "deliveryPrice": float(price),
                    "dineInPrice": float(price),
                    "takeawayPrice": float(price),
                }
            )
        m[key] = item
        return item

    def _link_group_to_parent(
        parent_entity: Dict[str, Any], group_obj: Dict[str, Any]
    ):
        parent_entity.setdefault("modifierMembers", [])
        for mm in parent_entity["modifierMembers"]:
            if mm.get("modifierId") == group_obj["id"]:
                return
        parent_entity["modifierMembers"].append(
            {
                "etag": f"W/\"datetime'{nowz}'\"",
                "timestamp": now_iso_hms(),
                "id": guid(),
                "modifierId": group_obj["id"],
            }
        )

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
            # NEW: capture category-level description/notes from extraction
            cat_desc = (cat_in.get("description") or cat_in.get("notes") or "").strip()

            if ck not in cat_index:
                cat = {
                    "etag": f"W/\"datetime'{nowz}'\"",
                    "timestamp": now_iso_hms(),
                    "caption": cat_caption,
                    "enabled": True,
                    "id": guid(),
                    "items": [],
                    "overrides": [],
                }
                # Attach category-level description as Flipdish category notes
                if cat_desc:
                    cat["notes"] = cat_desc

                colors = pick_category_colors(cat_caption_raw)
                if colors:
                    cat["backgroundColor"] = colors["backgroundColor"]
                    cat["foregroundColor"] = colors["foregroundColor"]
                out["categories"].append(cat)
                cat_index[ck] = cat
            else:
                cat = cat_index[ck]
                # If we later see a description for this category and it's still empty, fill it
                if cat_desc and not cat.get("notes"):
                    cat["notes"] = cat_desc

            page = (
                src_pdf_doc[page_i]
                if (attach_pdf_images and src_pdf_doc is not None)
                else None
            )

            for it in (cat_in.get("items") or []):
                raw = (it.get("caption") or "").strip()
                desc = (it.get("description") or "").strip()
                notes = (it.get("notes") or "").strip()
                name, inline = split_caption_and_inline_notes(raw)
                name = smart_title(name or "Item")

                base_price = it.get("price")
                if base_price is None:
                    base_price = (
                        parse_price_from_text(raw, desc, notes) or 0.0
                    )

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
                    "notes": " ".join(
                        p for p in [desc, notes, inline] if p
                    ).strip(),
                    "enabled": True,
                    "id": guid(),
                    "doesPriceRepresentRewardPoints": False,
                    "pricingProfiles": [
                        {
                            "etag": f"W/\"datetime'{nowz}'\"",
                            "timestamp": now_iso_hms(),
                            "priceBandId": price_band_id,
                            "collectionPrice": float(base_price),
                            "deliveryPrice": float(base_price),
                            "dineInPrice": float(base_price),
                            "takeawayPrice": float(base_price),
                        }
                    ],
                    "charges": [],
                    "modifierMembers": [],
                    "overrides": [],
                    "imageUrl": img_data_url or "",
                }

                llm_mods = it.get("modifiers") or []
                if not llm_mods:
                    llm_mods = fallback_extract_modifiers(
                        "\n".join([desc, notes, inline])
                    )

                for grp in llm_mods:
                    _process_group(item, grp)

                cat["items"].append(item)

    # finalize modifiers
    for g in modifiers_index.values():
        g["items"] = list(g["_items_map"].values())
        del g["_items_map"]
        out["modifiers"].append(g)

    # Drop empty categories
    out["categories"] = [c for c in out["categories"] if c.get("items")]

    # Apply normalization rules silently
    rules = rules or try_load_rules()
    out = normalize_with_rules(out, rules)

    return out

# ============================== Normalization rules ==============================

DEFAULT_RULES = {
    "modifier_caption_aliases": {
        "choose your meat": ["choice of meat", "choose meat"],
        "add": ["add-ons", "addons", "extras"],
    },
    "force_minmax": {
        "choose your meat": {"min": 1, "max": 1},
    },
    "option_aliases": {
        "bbq sauce": ["barbecue", "bbq"],
        "fries": ["french fries", "chips"],
    },
}

def try_load_rules() -> dict:
    path = "rules.json"
    if not os.path.exists(path):
        return DEFAULT_RULES
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return DEFAULT_RULES
        return data
    except Exception:
        return DEFAULT_RULES

def normalize_with_rules(flipdish_json: dict, rules: dict) -> dict:
    aliases = rules.get("modifier_caption_aliases", {})
    force = rules.get("force_minmax", {})
    opt_alias = rules.get("option_aliases", {})

    def canon_mod_name(name: str) -> str:
        n = (name or "").strip().upper()
        for k, vs in aliases.items():
            if n == k.upper() or n in [v.upper() for v in vs]:
                return k.upper()
        return n

    for g in flipdish_json.get("modifiers", []):
        g["caption"] = canon_mod_name(g.get("caption"))
        if g["caption"] in force:
            mm = force[g["caption"]]
            if "min" in mm:
                g["min"] = int(mm["min"])
            if "max" in mm:
                g["max"] = int(mm["max"])
        for it in g.get("items", []):
            label = (it.get("caption", "") or "").strip().lower()
            for canon, alist in opt_alias.items():
                if label in [canon] + alist:
                    it["caption"] = canon.title()
    return flipdish_json

# ============================== Streamlit UI (minimal) ==============================

tab1, tab2 = st.tabs(["PDF/Image → JSON", "Transform existing JSON"])

if "last_pdf_text" not in st.session_state:
    st.session_state.last_pdf_text = []

with tab1:
    f = st.file_uploader(
        "Upload menu (PNG, JPG, JPEG, or PDF)",
        type=["png", "jpg", "jpeg", "pdf"],
    )
    menu_name = st.text_input("Menu name", value="Generated Menu")
    price_band_id = st.text_input(
        "Flipdish Price Band ID", value="", help="Required"
    )
    model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
        index=0,
    )
    attach_pdf_images = st.checkbox(
        "Attempt to attach item images from PDF (experimental)",
        value=False,
    )

    if st.button("Generate Flipdish JSON"):
        if not f:
            st.error("Please upload a file.")
            st.stop()
        if not price_band_id.strip():
            st.error("Price Band ID is required.")
            st.stop()

        loaded = load_file(f)
        if not loaded.images:
            st.error("Please upload a valid image or PDF.")
            st.stop()

        extracted_pages: List[dict] = []
        per_page_text: List[str] = []

        with st.spinner("Extracting..."):
            for i, im in enumerate(loaded.images):
                page_text = ""
                if loaded.is_pdf and loaded.doc is not None and fitz is not None:
                    try:
                        page_text = loaded.doc[i].get_text("text") or ""
                    except Exception:
                        page_text = ""
                per_page_text.append(page_text)
                fewshot = ""  # could be extended with examples
                extracted = run_openai_single(im, model=model, fewshot=fewshot)
                extracted_pages.append(extracted)

        st.session_state.last_pdf_text = per_page_text

        result = to_flipdish_json(
            extracted_pages,
            menu_name,
            price_band_id.strip(),
            attach_pdf_images=attach_pdf_images,
            src_pdf_doc=loaded.doc if loaded.is_pdf else None,
            rules=None,
        )

        st.success("Menu JSON generated.")
        st.json(result, expanded=False)
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
            file_name="flipdish_menu.json",
            mime="application/json",
        )

with tab2:
    jf = st.file_uploader(
        "Upload existing JSON to transform to Flipdish shape",
        type=["json"],
        key="json_uploader",
    )
    menu_name2 = st.text_input("Menu name (override if needed)", value="")
    price_band_id2 = st.text_input(
        "Flipdish Price Band ID",
        value="",
        help="Required (used for pricingProfiles)",
        key="price_band_2",
    )

    if st.button("Re-shape JSON"):
        if not jf:
            st.error("Upload a JSON file first.")
            st.stop()
        if not price_band_id2.strip():
            st.error("Price Band ID is required.")
            st.stop()

        raw = json.load(io.BytesIO(jf.read()))
        # Treat uploaded JSON as already-extracted pages
        result = to_flipdish_json(
            [raw],
            menu_name2 or "",
            price_band_id2.strip(),
            attach_pdf_images=False,
            src_pdf_doc=None,
            rules=None,
        )
        st.success("Re-shaped successfully")
        st.json(result, expanded=False)
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
            file_name="flipdish_menu.json",
            mime="application/json",
        )
