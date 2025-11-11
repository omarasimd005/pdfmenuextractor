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
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import requests
from PIL import Image, ImageFile

import streamlit as st
from dotenv import load_dotenv

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

ImageFile.LOAD_TRUNCATED_IMAGES = True
load_dotenv()
st.set_page_config(page_title="Flipdish Menu Builder", layout="wide")


# ---------- Utilities ----------

def guid() -> str:
    return str(uuid.uuid4())


def now_iso_z() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def now_iso_hms() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat()


def is_price_token(tok: str) -> bool:
    return bool(re.search(r"\d", tok)) and any(ch.isdigit() for ch in tok)


def parse_price_from_text(*chunks: str) -> Optional[float]:
    text = " ".join([c for c in chunks if c]) or ""
    text = text.replace(",", "")
    m = re.search(r"([-+]?\d+(\.\d{1,2})?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def smart_title(text: str) -> str:
    if not text:
        return text
    SMALL_WORDS = {
        "a", "an", "the", "and", "but", "or", "for", "nor",
        "as", "at", "by", "for", "from", "in", "into", "near",
        "of", "on", "onto", "to", "vs", "via", "with"
    }

    def _cap_word(w: str) -> str:
        if not w:
            return w
        if w.lower() in SMALL_WORDS:
            return w.lower()
        if w.isupper() and len(w) > 1:
            return w
        return w[0].upper() + w[1:].lower()

    def _cap_hyphenated(token: str) -> str:
        parts = token.split("-")
        return "-".join(_cap_word(p) for p in parts)

    if text.isupper():
        return text

    tokens = re.split(r'(\s+)', text.strip())
    words_only = [t for t in tokens if not re.match(r'\s+', t)]
    if not words_only:
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
            if (word_index == 0 or
                word_index == len(words_only) - 1 or
                lower not in SMALL_WORDS):
                out = base[0].upper() + base[1:]
            else:
                out = base
        result.append(out)
        word_index += 1
    return "".join(result)


def split_caption_and_inline_notes(caption: str) -> Tuple[str, str]:
    if not caption:
        return "", ""
    m = re.match(r"^(.*?)(?:[-–—:]+\s*)(.*)$", caption.strip())
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        if right and " " in right:
            return left, right
    return caption.strip(), ""


def pick_category_colors(caption: str) -> Optional[Dict[str, str]]:
    if not caption:
        return None
    low = caption.lower()
    if any(k in low for k in ["breakfast", "brunch", "morning"]):
        return {"backgroundColor": "#FFF7E6", "foregroundColor": "#5A3E1B"}
    if any(k in low for k in ["burger", "grill", "bbq"]):
        return {"backgroundColor": "#FFF0F0", "foregroundColor": "#7A1F1F"}
    if any(k in low for k in ["pizza"]):
        return {"backgroundColor": "#FFF5EB", "foregroundColor": "#6B2E11"}
    if any(k in low for k in ["drinks", "beverages", "smoothies"]):
        return {"backgroundColor": "#E8F7FF", "foregroundColor": "#0A3A5A"}
    if any(k in low for k in ["dessert", "sweet"]):
        return {"backgroundColor": "#FFF0FA", "foregroundColor": "#6A174F"}
    return None


# ---------- Few-shot + Rules ----------

def load_examples() -> List[dict]:
    path = "examples.jsonl"
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


EXAMPLES_CACHE = load_examples()


def top_k_examples(query_text: str, k: int = 3) -> List[dict]:
    if not EXAMPLES_CACHE:
        return []
    q = (query_text or "").lower()
    def score(ex):
        src = (ex.get("source", "") or "").lower()
        fj = json.dumps(ex.get("flipdish", {}), ensure_ascii=False).lower()
        s = 0
        for tok in set(re.findall(r"[a-z0-9]+", q)):
            if tok in src:
                s += 2
            if tok in fj:
                s += 1
        return s
    ranked = sorted(EXAMPLES_CACHE, key=score, reverse=True)
    return [r for r in ranked if score(r) > 0][:k]


def build_fewshot_context(query_text: str) -> str:
    shots = top_k_examples(query_text, k=3)
    if not shots:
        return ""
    lines = []
    for ex in shots:
        lines.append(json.dumps({
            "source_excerpt": (ex.get("source", "") or "")[:400],
            "expected_flipdish_piece": ex.get("flipdish", {})
        }, ensure_ascii=False))
    return "\n".join(lines)


def try_load_rules() -> dict:
    path = "rules.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "modifier_caption_aliases": {
            "ADD ONS": "ADD ONS",
            "ADD ONS:": "ADD ONS",
            "EXTRAS": "EXTRAS",
            "SIDES": "SIDES",
        },
        "force_minmax": {},
        "option_aliases": {}
    }


# ---------- Prompt for LLM Extraction ----------

BASE_EXTRACTION_PROMPT = """
You output ONLY JSON (no markdown) with this schema:

{
  "name": string,
  "categories": [
    {
      "caption": string,
      "description": string,
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
- "Choice of toast or pancakes", "Choose from ..."
- Upgrades: "… (+2 to upgrade …)" as options with price
- CONDITIONALS: If a choice leads to another selection (e.g., similar to "If Waffles then choose syrup"), attach the follow-up group(s) under that option's "modifiers".

Rules:
- Item price numeric; ignore currency symbols.
- Options without explicit price -> price=null.
- Keep headings with a price as items; ignore decorative section headers.
- If there is descriptive text directly under a category heading that clearly describes the whole section (not just one item),
  put that text in the category's "description" field.
- Do NOT duplicate the same category description into every item.
- If no category-level description exists, use an empty string "" for "description".
"""


def _img_hash(img: Image.Image) -> str:
    return hashlib.blake2b(img.tobytes(), digest_size=16).hexdigest()


@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def _cached_extract_page(img_bytes: bytes, model: str, openai_api_key: str, system_prompt: str) -> dict:
    """
    Single-page vision+json extraction.
    """
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)

    b64 = base64.b64encode(img_bytes).decode("ascii")
    prompt = system_prompt

    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract Flipdish menu-structure JSON from this page."
                    },
                    {
                        "type": "input_image",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}"
                        }
                    }
                ]
            }
        ]
    )
    txt = resp.choices[0].message.content or "{}"
    try:
        return json.loads(txt)
    except Exception:
        try:
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1:
                return json.loads(txt[start:end + 1])
        except Exception:
            pass
    return {"categories": []}


def raster_pdf_to_images(file: bytes, max_pages: int = 30) -> List[Image.Image]:
    if fitz is None:
        return []
    pdf = fitz.open(stream=file, filetype="pdf")
    pages = []
    for i, page in enumerate(pdf):
        if i >= max_pages:
            break
        zoom = 300 / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    return pages


def nearest_image_crop(page, rect, max_search=3) -> Optional[bytes]:
    try:
        ir = page.get_images(full=True)
    except Exception:
        return None
    x0, y0, x1, y1 = rect
    center_y = (y0 + y1) / 2.0
    best = None
    best_dy = None
    for (xref, _, _, _, _, _, _, _) in ir[:100]:
        try:
            bbox = page.get_image_rects(xref)[0]
        except Exception:
            continue
        ix0, iy0, ix1, iy1 = bbox
        if ix1 < x0 or ix0 > x1:
            continue
        cy = (iy0 + iy1) / 2.0
        dy = abs(cy - center_y)
        if best_dy is None or dy < best_dy:
            best_dy = dy
            best = bbox
    if best is None or best_dy is None or best_dy > max_search * 20:
        return None
    ix0, iy0, ix1, iy1 = best
    rect = fitz.Rect(ix0, iy0, ix1, iy1)
    try:
        pix = page.get_pixmap(clip=rect, alpha=False)
        return pix.tobytes("png")
    except Exception:
        return None


def find_item_rects(page, name: str) -> List[Tuple[float, float, float, float]]:
    rects = []
    try:
        words = page.get_text("words")
    except Exception:
        return rects
    tokens = name.split()
    if not tokens:
        return rects
    first = tokens[0].lower()
    for (x0, y0, x1, y1, w, b, i, n) in words:
        if w.lower() == first:
            rects.append((x0, y0, x1, y1))
    return rects


# ---------- Fallback Modifiers ----------

def fallback_extract_modifiers(text: str) -> List[dict]:
    if not text:
        return []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    groups = []
    current = None

    def flush():
        nonlocal current
        if current and current.get("options"):
            groups.append(current)
        current = None

    for line in lines:
        m = re.match(r"^(add|extra|choice of|choose|choose from)\b[:\-\s]*(.*)$", line, re.I)
        if m:
            flush()
            caption = m.group(0).strip().rstrip(":")
            current = {
                "caption": smart_title(caption),
                "min": None,
                "max": None,
                "options": []
            }
            rest = m.group(2).strip()
            if rest:
                parts = re.split(r",|\bor\b|\band\b", rest)
                for p in parts:
                    p = p.strip()
                    if not p:
                        continue
                    price = parse_price_from_text(p)
                    name = re.sub(r"[-+]?\\s*\\d+(\\.\\d{1,2})?", "", p).strip()
                    current["options"].append({
                        "caption": smart_title(name),
                        "price": price
                    })
            continue

        pm = re.match(r"^(.+?)\\s*[+](\\d+(?:\\.\\d{1,2})?)$", line)
        if pm:
            if not current:
                current = {"caption": "Add Ons", "min": None, "max": None, "options": []}
            name = smart_title(pm.group(1).strip())
            price = float(pm.group(2))
            current["options"].append({"caption": name, "price": price})
            continue

    flush()
    return groups


# ---------- JSON → Flipdish ----------

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

    def ensure_group(caption: str,
                     min_sel: Optional[int] = None,
                     max_sel: Optional[int] = None,
                     can_repeat: Optional[bool] = None) -> Dict[str, Any]:
        key_raw = caption or "ADD"
        key = key_raw.strip().upper()
        nowh = now_iso_hms()
        if key in modifiers_index:
            return modifiers_index[key]
        g = {
            "etag": f"W/\"datetime'{nowz}'\"",
            "timestamp": nowh,
            "canSameItemBeSelectedMultipleTimes": (
                True if can_repeat is None else bool(can_repeat)
            ),
            "caption": smart_title(key_raw if key_raw else "ADD"),
            "enabled": True,
            "hiddenInOrderFlow": False,
            "id": guid(),
            "max": 1 if max_sel is None else max_sel,
            "min": 0 if min_sel is None else min_sel,
            "position": len(modifiers_index),
            "items": [],
            "overrides": []
        }
        g["_items_map"] = {}
        modifiers_index[key] = g
        return g

    def _ensure_option_item(group: Dict[str, Any],
                            oname: str,
                            price: Optional[float]) -> Dict[str, Any]:
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

    def _process_group(parent_entity: Dict[str, Any],
                       grp: Dict[str, Any]):
        cap = (grp.get("caption") or "Add Ons").strip()
        min_sel = grp.get("min")
        max_sel = grp.get("max")
        opts = grp.get("options") or []
        group = ensure_group(cap, min_sel, max_sel)
        for opt in opts:
            oname = smart_title((opt.get("caption") or "").strip())
            if not oname:
                continue
            price = opt.get("price")
            opt_item = _ensure_option_item(group, oname, price)
            for child_grp in (opt.get("modifiers") or []):
                _process_group(opt_item, child_grp)

        _link_group_to_parent(parent_entity, group)

    def _link_group_to_parent(parent_entity: Dict[str, Any],
                              group: Dict[str, Any]):
        mems = parent_entity.setdefault("modifierMembers", [])
        group_id = group["id"]
        for m in mems:
            if m.get("modifierGroupId") == group_id:
                return
        mems.append({
            "enabled": True,
            "id": guid(),
            "modifierGroupId": group_id,
            "position": len(mems),
        })

    cat_index: Dict[str, Any] = {}

    for page_i, data in enumerate(extracted_pages):
        # For this page, decide if we can use it to crop images
        page = src_pdf_doc[page_i] if (attach_pdf_images and src_pdf_doc is not None) else None

        for cat_in in (data.get("categories") or []):
            cat_caption_raw = (cat_in.get("caption") or "Category").strip()
            cat_caption = smart_title(cat_caption_raw).upper()
            ck = cat_caption.lower()

            # Category-level description / notes from extraction or existing JSON
            cat_desc = (cat_in.get("description") or "").strip()
            cat_notes_in = (cat_in.get("notes") or "").strip()
            cat_notes_val = cat_desc or cat_notes_in

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

                # Attach category description as Flipdish notes (if present)
                if cat_notes_val:
                    cat["notes"] = cat_notes_val

                colors = pick_category_colors(cat_caption_raw)
                if colors:
                    cat["backgroundColor"] = colors["backgroundColor"]
                    cat["foregroundColor"] = colors["foregroundColor"]

                out["categories"].append(cat)
                cat_index[ck] = cat
            else:
                cat = cat_index[ck]
                # If we later see a description for an already-seen category and it has no notes yet, fill it once.
                if cat_notes_val and not cat.get("notes"):
                    cat["notes"] = cat_notes_val

            # ---- Items for this category ----
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
        return aliases.get(n, n)

    for g in flipdish_json.get("modifiers", []):
        n0 = g.get("caption")
        n1 = canon_mod_name(n0)
        g["caption"] = smart_title(n1)
        if n1 in force:
            fm = force[n1]
            g["min"] = fm.get("min", g.get("min"))
            g["max"] = fm.get("max", g.get("max"))
        for it in g.get("items", []):
            for pp in it.get("pricingProfiles", []):
                for fld in ["collectionPrice", "deliveryPrice", "dineInPrice", "takeawayPrice"]:
                    v = pp.get(fld)
                    try:
                        pp[fld] = float(v)
                    except Exception:
                        pp[fld] = 0.0
            for mm in it.get("modifierMembers", []):
                pass
            for ch in it.get("charges", []):
                pass

    for cat in flipdish_json.get("categories", []):
        for item in cat.get("items", []):
            for mm in item.get("modifierMembers", []):
                pass
            for ch in item.get("charges", []):
                pass

    return flipdish_json


# ---------- Streamlit UI ----------

def main():
    st.title("Flipdish Menu Builder")

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    model = st.text_input("Model name", value="gpt-4.1-mini")
    price_band_id = st.text_input("Price Band ID", value="")
    attach_pdf_images = st.checkbox("Attach images from PDF near items", value=False)

    tab1, tab2 = st.tabs(["PDF / Image → Flipdish JSON", "Transform existing JSON"])

    with tab1:
        uploaded = st.file_uploader("Upload menu PDF or image(s)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=False)
        menu_name = st.text_input("Menu Name (optional)", "")
        if st.button("Generate JSON", disabled=not uploaded or not openai_api_key or not price_band_id):
            if not uploaded:
                st.error("Upload a file.")
                st.stop()
            if not openai_api_key:
                st.error("OpenAI API key required.")
                st.stop()
            if not price_band_id.strip():
                st.error("Price Band ID is required.")
                st.stop()

            bytes_data = uploaded.read()
            images = []
            src_pdf_doc = None
            if uploaded.type == "application/pdf":
                if fitz is None:
                    st.error("PyMuPDF not installed on server.")
                    st.stop()
                src_pdf_doc = fitz.open(stream=bytes_data, filetype="pdf")
                images = raster_pdf_to_images(bytes_data)
            else:
                try:
                    img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
                    images = [img]
                except Exception as e:
                    st.error(f"Failed to read image: {e}")
                    st.stop()

            fewshot = build_fewshot_context(uploaded.name)
            system_prompt = BASE_EXTRACTION_PROMPT + ("\nFew-shot examples:\n" + fewshot if fewshot else "")

            pages_data = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                page_json = _cached_extract_page(buf.getvalue(), model, openai_api_key, system_prompt)
                pages_data.append(page_json)

            result = to_flipdish_json(
                pages_data,
                menu_name or uploaded.name,
                price_band_id.strip(),
                attach_pdf_images,
                src_pdf_doc,
                rules=None,
            )

            st.success("Flipdish JSON generated")
            st.json(result, expanded=False)
            st.download_button(
                "Download JSON",
                data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
                file_name="flipdish_menu.json",
                mime="application/json"
            )

    with tab2:
        st.write("Upload an existing JSON (any structure) and we will reshape into Flipdish schema.")
        jf = st.file_uploader("Upload JSON", type=["json"], key="json_upload")
        menu_name2 = st.text_input("Menu Name override (optional)", key="menu2")
        price_band_id2 = st.text_input("Price Band ID (required)", key="pb2")
        if st.button("Transform JSON", disabled=not jf or not price_band_id2):
            if not jf:
                st.error("Upload a JSON file first.")
                st.stop()
            if not price_band_id2.strip():
                st.error("Price Band ID is required.")
                st.stop()
            raw = json.load(io.BytesIO(jf.read()))
            # For transform mode we treat uploaded JSON as extracted_pages-style list
            # or single object with categories.
            if isinstance(raw, dict) and "categories" in raw:
                pages_data = [raw]
            elif isinstance(raw, list):
                pages_data = raw
            else:
                pages_data = [{"categories": raw.get("categories", [])}] if isinstance(raw, dict) else []
            result = to_flipdish_json(
                pages_data,
                menu_name2 or "Transformed Menu",
                price_band_id2.strip(),
                False,
                None,
                rules=None
            )
            st.success("Re-shaped successfully")
            st.json(result, expanded=False)
            st.download_button(
                "Download JSON",
                data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
                file_name="flipdish_menu.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
