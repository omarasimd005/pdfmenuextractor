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
except ImportError:
    fitz = None

# ============================== Utils ==============================

def guid() -> str:
    return str(uuid.uuid4())


def now_iso_z() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def now_iso_hms() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat()


def to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def read_pdf(file_bytes: bytes) -> Optional["fitz.Document"]:
    if fitz is None:
        return None
    return fitz.open(stream=file_bytes, filetype="pdf")


def smart_title(text: str) -> str:
    if not text:
        return ""
    SMALL_WORDS = {
        "a", "an", "and", "as", "at", "but", "by",
        "for", "from", "in", "into", "of", "on",
        "or", "the", "to", "with"
    }

    def _cap_hyphenated(word: str) -> str:
        parts = word.split("-")
        return "-".join(p.capitalize() if p else p for p in parts)

    if len(text) < 4:
        return text.upper()

    if " " not in text and "-" not in text:
        if text.isupper():
            return text
        return text.capitalize()

    if text.isupper() and len(text) > 4:
        return text

    tokens = re.split(r'(\s+)', text.strip())
    words_only = [t for t in tokens if not re.match(r'\s+', t)]
    if not words_only:
        return text
    if len(words_only) > 12:
        return text
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
    (re.compile(r"pizza", re.I), "#D35400"),
    (re.compile(r"burger", re.I), "#8D6E63"),
    (re.compile(r"pasta", re.I), "#AF601A"),
    (re.compile(r"grill", re.I), "#B71C1C"),
    (re.compile(r"breakfast", re.I), "#F39C12"),
]

DEFAULT_CATEGORY_COLOR = "#607D8B"


def category_color(name: str) -> str:
    for rx, color in CATEGORY_COLOR_RULES:
        if rx.search(name or ""):
            return color
    return DEFAULT_CATEGORY_COLOR


# ============================== Simple price parsing ==============================

PRICE_RE = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d{1,2})?)\s*(?:[€£$])?")


def parse_price_from_text(*texts: str) -> Optional[float]:
    for t in texts:
        if not t:
            continue
        m = PRICE_RE.search(t)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


# ============================== Example learning store ==============================

EXAMPLES_PATH = "examples.jsonl"


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


# ============================== PDF helpers ==============================

def find_item_rects(page: "fitz.Page", name: str) -> List["fitz.Rect"]:
    rects = []
    if not name:
        return rects
    text_instances = page.search_for(name, hit_max=16)
    for inst in text_instances:
        rects.append(inst)
    return rects


def nearest_image_crop(page: "fitz.Page", around: "fitz.Rect", margin: int = 10) -> Optional[bytes]:
    images = page.get_images(full=True)
    if not images:
        return None
    best = None
    best_dist = None
    for img in images:
        xref = img[0]
        try:
            rects = page.get_image_bbox(xref)
        except Exception:
            continue
        if not rects:
            continue
        r = rects
        cx = (r.x0 + r.x1) / 2
        cy = (r.y0 + r.y1) / 2
        ax = (around.x0 + around.x1) / 2
        ay = (around.y0 + around.y1) / 2
        d = (cx - ax) ** 2 + (cy - ay) ** 2
        if best is None or d < best_dist:
            best = r
            best_dist = d
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
            pm = PLUS_PRICE_LINE.match(t)
            if pm:
                nm = pm.group("name").strip()
                try:
                    pr = float(pm.group("price"))
                except Exception:
                    pr = None
            else:
                nm = t
                pr = None
            if not nm:
                continue
            groups.setdefault(gcap, []).append((nm, pr))

    for m in ADD_PATTERN.finditer(text):
        seg = m.group("opts")
        pr = m.group("price")
        try:
            prf = float(pr) if pr else None
        except Exception:
            prf = None
        for opt in split_option_list(seg):
            groups.setdefault("ADD", []).append((opt, prf))

    for m in CHOICE_PATTERN.finditer(text):
        seg = m.group("opts")
        for opt in split_option_list(seg):
            groups.setdefault("CHOOSE", []).append((opt, None))

    out = []
    for cap, items in groups.items():
        seen, opts = set(), []
        for n, p in items:
            key = (n.lower(), p if p is not None else -1)
            if key in seen:
                continue
            seen.add(key)
            opts.append({"caption": n, "price": p})
        if opts:
            out.append({"caption": cap, "min": None, "max": None, "options": opts})
    return out


# ============================== Allergen detection (added) ==============================

# Canonical allergen labels supported for Flipdish dietaryConfiguration.dietaryTags
_ALLERGEN_DEFS = [
    ("Celery", [r"\bcelery\b"]),
    ("Crustaceans", [r"\bcrab\b", r"\blobster\b", r"\bprawn", r"\bshrimp", r"\bcrustacean"]),
    ("Egg", [r"\begg\b", r"\beggs\b", r"\bmayonnaise\b"]),
    ("Fish", [r"\bfish\b", r"\bsalmon\b", r"\btuna\b", r"\btrout\b"]),
    ("Gluten", [r"\bgluten\b"]),
    ("Lupin", [r"\blupin\b"]),
    ("Milk", [r"\bmilk\b", r"\bcheese\b", r"\bbutter\b", r"\bcream\b", r"\byogurt\b"]),
    ("Molluscs", [r"\bmussel", r"\boyster", r"\bclam", r"\bscallop", r"\bmollusc"]),
    ("Mustard", [r"\bmustard\b"]),
    ("Nuts", [r"\bnuts?\b", r"\balmond", r"\bwalnut", r"\bpecan", r"\bhazelnut", r"\bcashew", r"\bpistachio"]),
    ("Peanuts", [r"\bpeanut", r"\bgroundnut"]),
    ("Sesame", [r"\bsesame\b", r"\btahini\b"]),
    ("Soya", [r"\bsoya\b", r"\bsoy\b", r"\bsoybean"]),
    ("Soybeans", [r"\bsoybean", r"\bsoybeans"]),
    ("Sulphur Dioxide", [r"\bsulphur\s*dioxide\b", r"\bsulfur\s*dioxide\b",
                         r"\bsulphites?\b", r"\bsulfites?\b"]),
    ("Wheat", [r"\bwheat\b", r"\bsemolina\b"]),
    ("Alcohol", [r"\balcohol\b", r"\bbrandy\b", r"\bwhiskey\b", r"\bvodka\b",
                 r"\brum\b", r"\bgin\b", r"\bwine\b", r"\bbeer\b"])
]


def _detect_allergens(text: str) -> list:
    """Heuristic allergen detector from free text.
    Returns canonical labels; does not affect any other logic.
    """
    if not text:
        return []
    t = text.lower()
    found = []
    for label, patterns in _ALLERGEN_DEFS:
        for pat in patterns:
            if re.search(pat, t):
                found.append(label)
                break
    # De-dupe, stable order
    seen = set()
    out = []
    for lbl in found:
        if lbl not in seen:
            seen.add(lbl)
            out.append(lbl)
    return out


def _merge_allergen_params(existing_params_json: str or None, allergens: list) -> str:
    """Merge allergens into paramsJson.dietaryConfiguration.dietaryTags.

    - Preserves any existing paramsJson structure.
    - De-duplicates dietaryTags.
    - Returns JSON string for paramsJson (or original if nothing to add).
    """
    if not allergens:
        return existing_params_json or ""
    base = {}
    if existing_params_json:
        try:
            base = json.loads(existing_params_json)
            if not isinstance(base, dict):
                base = {}
        except Exception:
            base = {}
    dc = base.get("dietaryConfiguration") or {}

    existing = []
    dt = dc.get("dietaryTags")
    if isinstance(dt, str):
        for part in dt.split(","):
            p = part.strip()
            if p:
                existing.append(p)

    merged = []
    seen = set()
    for v in existing + allergens:
        if v and v not in seen:
            seen.add(v)
            merged.append(v)

    if merged:
        dc["dietaryTags"] = ",".join(merged)
        base["dietaryConfiguration"] = dc

    if not base:
        return existing_params_json or ""
    return json.dumps(base, ensure_ascii=False)


def _extract_allergens_from_item_src(src_item: dict, combined_text: str) -> list:
    """Pull allergens from any structured field (if present) + nearby free text."""
    tags = []

    # Structured field from the extractor, if it ever exists (forwards-compatible)
    ai = src_item.get("allergens")
    if isinstance(ai, list):
        for entry in ai:
            if isinstance(entry, str):
                tags.extend(_detect_allergens(entry))
    elif isinstance(ai, str):
        tags.extend(_detect_allergens(ai))

    # Also detect from combined visible text
    tags.extend(_detect_allergens(combined_text))

    # De-dupe stable
    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
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
        "scheduleConfigs": [],
        "categories": [],
        "modifiers": []
    }

    modifiers_index: Dict[str, Any] = {}

    def ensure_group(caption: str, min_sel: Optional[int], max_sel: Optional[int],
                     can_repeat: Optional[bool]) -> Dict[str, Any]:
        key = caption.strip().lower()
        if key in modifiers_index:
            return modifiers_index[key]
        g = {
            "etag": f"W/\"datetime'{nowz}'\"",
            "timestamp": now_iso_hms(),
            "id": guid(),
            "caption": caption,
            "min": 0 if min_sel is None else min_sel,
            "max": max_sel,
            "canSameItemBeSelectedMultipleTimes": True if can_repeat is None else can_repeat,
            "items": []
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

        # auto-tag modifier-level allergens from option caption text only
        opt_allergens = _detect_allergens(oname)
        if opt_allergens:
            opt_item["paramsJson"] = _merge_allergen_params(
                opt_item.get("paramsJson"),
                opt_allergens
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
                    "id": guid(),
                    "caption": cat_caption,
                    "color": category_color(cat_caption),
                    "items": []
                }
                cat_index[ck] = cat
                out["categories"].append(cat)
            else:
                cat = cat_index[ck]

            page = src_pdf_doc[page_i] if (attach_pdf_images and src_pdf_doc is not None) else None

            for it in (cat_in.get("items") or []):
                raw = (it.get("caption") or "").strip()
                desc = (it.get("description") or "").strip()
                notes = (it.get("notes") or "").strip()
                name_inline = raw
                name = raw
                inline = ""

                # Existing logic for title vs inline notes
                m_inline = re.search(r"\(([^)]+)\)\s*$", raw)
                if m_inline:
                    inline = m_inline.group(1).strip()
                    name = raw[:m_inline.start()].strip()
                else:
                    name = raw

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

                # auto-tag item-level allergens from surrounding text / structured hints
                item_allergen_text = " ".join([
                    raw or "",
                    desc or "",
                    notes or "",
                    inline or "",
                ])
                item_allergens = _extract_allergens_from_item_src(it, item_allergen_text)
                if item_allergens:
                    item["paramsJson"] = _merge_allergen_params(
                        item.get("paramsJson"),
                        item_allergens
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


# ============================== Normalization ==============================

def try_load_rules() -> dict:
    path = "rules.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def normalize_with_rules(flipdish_json: dict, rules: dict) -> dict:
    # Placeholder for silent tweaks; existing behavior untouched
    if not isinstance(rules, dict):
        return flipdish_json
    return flipdish_json


# ============================== Streamlit UI ==============================

st.set_page_config(page_title="Flipdish Menu Builder", layout="wide")

st.title("Flipdish Menu Builder")

tab1, tab2 = st.tabs(["From Scraped JSON", "From Raw Flipdish-style JSON"])

with tab1:
    st.header("1) Upload scraper output JSON")
    uploaded = st.file_uploader("Scraper JSON", type=["json"], key="scraper_json")
    menu_name = st.text_input("Menu name", value="Menu")
    price_band_id = st.text_input("Flipdish Price Band ID", value="")
    attach_images = st.checkbox("Attempt to auto-crop item images from original PDF (if available)", value=False)
    pdf_file = st.file_uploader("Optional: original menu PDF (for image crops)", type=["pdf"], key="menu_pdf")

    if st.button("Build Flipdish JSON from scraper output"):
        if not uploaded:
            st.error("Upload the scraper JSON first.")
            st.stop()
        if not price_band_id.strip():
            st.error("Price Band ID is required.")
            st.stop()

        scraped = json.load(io.BytesIO(uploaded.read()))

        pdf_doc = None
        if attach_images and pdf_file is not None:
            if fitz is None:
                st.warning("PyMuPDF not installed; cannot attach PDF images.")
            else:
                try:
                    pdf_doc = read_pdf(pdf_file.read())
                except Exception:
                    pdf_doc = None

        result = to_flipdish_json(
            [scraped],
            menu_name or "",
            price_band_id.strip(),
            attach_images and (pdf_doc is not None),
            pdf_doc,
            rules=None,
        )

        st.success("Menu generated")
        st.json(result, expanded=False)
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
            file_name="flipdish_menu.json",
            mime="application/json",
        )

with tab2:
    st.header("2) Re-shape already structured JSON")
    jf = st.file_uploader("Existing Flipdish-style JSON", type=["json"], key="flipdish_json")
    menu_name2 = st.text_input("Menu name override (optional)")
    price_band_id2 = st.text_input("Flipdish Price Band ID", value="", key="pb2")

    if st.button("Re-shape JSON"):
        if not jf:
            st.error("Upload a JSON file first.")
            st.stop()
        if not price_band_id2.strip():
            st.error("Price Band ID is required.")
            st.stop()

        raw = json.load(io.BytesIO(jf.read()))
        result = to_flipdish_json([raw], menu_name2 or "", price_band_id2.strip(), False, None, rules=None)
        st.success("Re-shaped successfully")
        st.json(result, expanded=False)
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
            file_name="flipdish_menu.json",
            mime="application/json",
        )
