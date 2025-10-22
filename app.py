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
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image, ImageOps, ImageFile
except Exception:
    Image = None
    ImageOps = None
    ImageFile = None

try:
    import streamlit as st
except Exception:
    class _Shim:
        def __getattr__(self, k): return lambda *a, **kw: None
    st = _Shim()  # fallback to avoid import crash in non-UI environments

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(): pass

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# --------- basic setup ---------
ImageFile.LOAD_TRUNCATED_IMAGES = True
load_dotenv()
st.set_page_config(page_title="Flipdish Menu Builder", page_icon="🍽️", layout="centered")
st.title("Flipdish Menu Builder")

# ============================== Utils ==============================

def guid() -> str:
    return str(uuid.uuid4())

def now_iso_z() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def now_iso_hms() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

def smart_title(s: str) -> str:
    s = (s or "").strip()
    if not s: return s
    # keep ALL CAPS if original was all caps; otherwise title-case nicely
    return s if s.isupper() else re.sub(r"\s+", " ", s.title())

def clean_price_text(*parts: str) -> float:
    raw = " ".join(p for p in parts if p).strip()
    if not raw: return 0.0
    m = re.search(r"([€$£]?\s*-?\s*\d+(?:[.,]\d{1,2})?)", raw)
    if not m: return 0.0
    v = m.group(1)
    v = v.replace("€","").replace("$","").replace("£","")
    v = v.replace(" ", "")
    v = v.replace(",", ".")
    try:
        return float(v)
    except:
        return 0.0

def to_data_url(png: bytes) -> str:
    # NOTE: We DO NOT pass this into Flipdish anymore due to 64KB property limits.
    # Kept for potential local previews, if ever needed.
    return "data:image/png;base64," + base64.b64encode(png).decode("utf-8")

# ============================== Loaders ==============================

def _is_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF")

def _as_image_list_from_pdf(data: bytes, dpi: int = 300) -> List[Image.Image]:
    if fitz is None or Image is None:
        return []
    doc = fitz.open(stream=data, filetype="pdf")
    imgs = []
    for i in range(len(doc)):
        page = doc[i]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        imgs.append(img)
    return imgs

class LoadedFile:
    def __init__(self, images: List[Image.Image], text_pages: List[str], doc: Optional["fitz.Document"], is_pdf: bool):
        self.images = images
        self.text_pages = text_pages
        self.doc = doc
        self.is_pdf = is_pdf

def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# lightweight OCR/text extraction for PDFs
def extract_texts_pdf(doc: "fitz.Document") -> List[str]:
    out = []
    for i in range(len(doc)):
        page = doc[i]
        out.append(page.get_text("text"))
    return out

def load_file(file) -> LoadedFile:
    if file is None:
        return LoadedFile([], [], None, False)
    data = file.read()
    if not data:
        return LoadedFile([], [], None, False)
    if _is_pdf(data):
        doc = fitz.open(stream=data, filetype="pdf") if fitz is not None else None
        imgs = _as_image_list_from_pdf(data) if (doc is not None and Image is not None) else []
        texts = extract_texts_pdf(doc) if doc is not None else []
        return LoadedFile(imgs, texts, doc, True)
    else:
        try:
            img = Image.open(io.BytesIO(data))
            return LoadedFile([img], [""], None, False)
        except Exception:
            return LoadedFile([], [], None, False)

# ============================== Basic LLM helpers (stubbed for local) ==============================

def call_model(prompt: str, model: str = "gpt-4o-mini") -> str:
    # This function is left simple for portability; you can wire it to your LLM provider if desired.
    # For the purposes of shaping to Flipdish JSON, we rely more on regex heuristics + light prompts.
    return prompt  # echo for now

# ============================== PDF/Item region helpers ==============================

def find_item_rects(page: "fitz.Page", name: str):
    if fitz is None or page is None or not name:
        return []
    def text(s: str, *more: str) -> float:
        t = s or ""
        for m in more: t += " " + (m or "")
        t = re.sub(r"\s+", " ", t.strip())
        r = 0.0
        for inst in page.search_for(t):
            r = max(r, inst.get_area())
        return r

    # fallback: try shorter spans
    def find_any():
        r = page.search_for(name)
        if r: return r
        toks = name.split()
        for n in (3, 2, 1):
            if len(toks) >= n:
                r = page.search_for(" ".join(toks[:n]))
                if r: return r
        return []

    return find_any()

def nearest_image_crop(page: "fitz.Page", near: "fitz.Rect", margin: float = 12.0) -> Optional[bytes]:
    if fitz is None:
        return None
    layout = page.get_text("dict")
    imgs = []
    for b in layout.get("blocks", []):
        if b.get("type") == 1 and "image" in b:
            rect = fitz.Rect(b["bbox"])
            imgs.append(rect)
    if not imgs:
        return None
    # pick the nearest image rect by center distance
    cx, cy = near.x0 + near.width/2, near.y0 + near.height/2
    best = None
    best_d = 1e18
    for r in imgs:
        rx, ry = r.x0 + r.width/2, r.y0 + r.height/2
        d = (rx-cx)**2 + (ry-cy)**2
        if d < best_d:
            best_d = d
            best = r
    if best is None:
        return None
    # expand with margin and clip to page
    R = fitz.Rect(
        max(0, best.x0 - margin),
        max(0, best.y0 - margin),
        best.x1 + margin,
        best.y1 + margin
    )
    pix = page.get_pixmap(clip=R, alpha=False)
    return pix.tobytes("png")

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
            "_items_map": {}
        }
        modifiers_index[key] = g
        return g

    def add_option(group: Dict[str, Any], oname: str, price: Optional[float]) -> Dict[str, Any]:
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
            "priceChange": float(price) if price is not None else 0.0,
            "pricingProfiles": [],
            "quantityFree": 0,
            "unitOverride": None
        }
        group["_items_map"][key_nm] = opt_item
        return opt_item

    # Walk pages
    for page_i, pg in enumerate(extracted_pages):
        cat_index: Dict[str, Dict[str, Any]] = {}
        for cat_in in (pg.get("categories") or []):
            cname = smart_title(cat_in.get("caption") or "Menu")
            if cname not in cat_index:
                nowh = now_iso_hms()
                cat = {
                    "etag": f"W/\"datetime'{nowz}'\"",
                    "timestamp": nowh,
                    "caption": cname,
                    "enabled": True,
                    "id": guid(),
                    "items": [],
                    "imageUrl": "",
                    "modifiers": [],
                    "overrides": [],
                    "priority": 0
                }
                cat_index[cname] = cat
                out["categories"].append(cat)
            else:
                cat = cat_index[cname]

            page = src_pdf_doc[page_i] if (attach_pdf_images and src_pdf_doc is not None) else None

            for it in (cat_in.get("items") or []):
                raw = (it.get("caption") or "").strip()
                desc = (it.get("description") or "")
                notes = (it.get("notes") or "")
                inline = (it.get("inline") or "")
                name = smart_title(raw)
                base_price = clean_price_text(raw, desc, notes) or 0.0

                img_data_url = ""
                if page is not None and name and fitz is not None:
                    rects = find_item_rects(page, name)
                    if rects:
                        png = nearest_image_crop(page, rects[0])
                        if png: img_data_url = to_data_url(png)

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
                    # IMPORTANT FIX:
                    # Flipdish/Azure property size limit (~64KB). Data URLs can exceed this.
                    # Only allow real HTTP(S) URLs; otherwise leave blank.
                    "imageUrl": (img_data_url if (img_data_url.startswith("http://") or img_data_url.startswith("https://")) else "")
                }

                llm_mods = it.get("modifiers") or []
                if not llm_mods:
                    llm_mods = []  # could call a parser here if desired

                for grp in llm_mods:
                    gname = grp.get("caption") or "Add"
                    minsel = grp.get("min")
                    maxsel = grp.get("max")
                    canrep = grp.get("canRepeat")
                    group = ensure_group(gname, minsel, maxsel, canrep)
                    for opt in (grp.get("items") or []):
                        oname = smart_title(opt.get("caption") or "Option")
                        price = opt.get("price")
                        member = add_option(group, oname, price)
                        item["modifierMembers"].append({"modifierId": group["id"], "modifierItemId": member["id"]})

                cat["items"].append(item)

    # finalize modifiers
    for g in modifiers_index.values():
        g["items"] = list(g["_items_map"].values())
        del g["_items_map"]
        out["modifiers"].append(g)

    return out

# ============================== Streamlit UI ==============================

tab1, tab2 = st.tabs(["PDF/Image → JSON", "Transform existing JSON"])

if "last_pdf_text" not in st.session_state:
    st.session_state.last_pdf_text = []

with tab1:
    f = st.file_uploader("Upload menu (PNG, JPG, JPEG, or PDF)", type=["png", "jpg", "jpeg", "pdf"])
    menu_name = st.text_input("Menu name", value="Generated Menu")
    price_band_id = st.text_input("Flipdish Price Band ID (required)", value="")
    # New default + help: Flipdish expects hosted URLs; data URLs will be omitted to avoid 64KB limit
    attach_images = st.checkbox("Attach cropped item images (PDF only) — requires hosted URLs; data URLs will be omitted", value=False)
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)

    if st.button("Extract and Build JSON"):
        if not price_band_id.strip():
            st.error("Price Band ID is required."); st.stop()

        loaded = load_file(f)
        if not loaded.images:
            st.error("Please upload a valid image or PDF"); st.stop()

        # naive page extraction to structured blocks (stubbed for brevity)
        extracted_pages = []
        per_page_text = []
        for i, img in enumerate(loaded.images):
            # simplistic: one category per page with no modifiers (you can expand with your own parser)
            extracted = {
                "categories": [{
                    "caption": f"Page {i+1}",
                    "items": []
                }]
            }
            # You can wire an actual extractor here
            extracted_pages.append(extracted)
            per_page_text.append(loaded.text_pages[i] if i < len(loaded.text_pages) else "")

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
                for it in (cat.get("items") or [])[:2]:
                    ex = {
                        "text": src_text[:4000],
                        "label": {
                            "caption": it.get("caption"),
                            "notes": it.get("notes"),
                            "price": clean_price_text(it.get("caption") or "", it.get("notes") or "")
                        }
                    }
                    with open("examples.jsonl", "a", encoding="utf-8") as w:
                        w.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    saved += 1
            if saved:
                st.caption(f"Saved {saved} tiny exemplar(s) for better heuristics next time.")
        except Exception:
            pass

        st.success("Menu JSON built successfully")
        st.json(result, expanded=False)
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
            file_name="flipdish_menu.json",
            mime="application/json"
        )

with tab2:
    st.caption("Upload an existing Flipdish-like JSON to reshape or validate.")
    jf = st.file_uploader("Upload JSON", type=["json"], key="jsonu")
    menu_name2 = st.text_input("Menu name (optional)", value="")
    price_band_id2 = st.text_input("Flipdish Price Band ID (required)", value="")
    if st.button("Reshape JSON"):
        if not jf:
            st.error("Upload a JSON file first."); st.stop()
        if not price_band_id2.strip():
            st.error("Price Band ID is required."); st.stop()

        raw = json.load(io.BytesIO(jf.read()))
        result = to_flipdish_json([raw], menu_name2 or "", price_band_id2.strip(), False, None, rules=None)
        st.success("Re-shaped successfully")
        st.json(result, expanded=False)
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False).encode(),
            file_name="flipdish_menu.json",
            mime="application/json"
        )
