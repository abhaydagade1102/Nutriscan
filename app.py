"""NutriScan - local nutrition scanner. Run: python app.py  ->  http://localhost:5000"""
import base64, difflib, json, os, re
import cv2, numpy as np, requests
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")
UA = {"User-Agent": "NutriScan/0.1 (abhaydagade1102@gmail.com)"}  # put your contact email
VISION_MODEL = os.getenv("VISION_MODEL", "claude-sonnet-5-5")

# ---------- additives DB: code -> (name, risk) ; risk: high / moderate / low ----------
ADDITIVES = {
    "102": ("Tartrazine", "high"), "110": ("Sunset Yellow", "high"), "122": ("Carmoisine", "high"),
    "124": ("Ponceau 4R", "high"), "129": ("Allura Red", "high"), "171": ("Titanium dioxide", "high"),
    "250": ("Sodium nitrite", "high"), "320": ("BHA", "high"), "321": ("BHT", "high"),
    "211": ("Sodium benzoate", "moderate"), "150d": ("Sulphite ammonia caramel", "moderate"),
    "338": ("Phosphoric acid", "moderate"), "407": ("Carrageenan", "moderate"),
    "621": ("Monosodium glutamate", "moderate"), "627": ("Disodium guanylate", "moderate"),
    "631": ("Disodium inosinate", "moderate"), "951": ("Aspartame", "moderate"),
    "950": ("Acesulfame K", "moderate"), "955": ("Sucralose", "moderate"),
    "330": ("Citric acid", "low"), "322": ("Lecithin", "low"), "471": ("Mono/diglycerides", "low"),
    "202": ("Potassium sorbate", "low"), "300": ("Ascorbic acid", "low"), "415": ("Xanthan gum", "low"),
}
NAME_INDEX = {v[0].lower(): k for k, v in ADDITIVES.items()}
NAME_INDEX.update({"msg": "621", "aspartame": "951", "tartrazine": "102", "sucralose": "955"})
PENALTY = {"high": 15, "moderate": 5, "low": 0}


def find_additives(text="", tags=()):
    """Match E/INS codes, exact names, and fuzzy names (OCR typos) -> list of additives."""
    found = set()
    for t in tags:  # Open Food Facts style: 'en:e330'
        m = re.search(r"e(\d{3}[a-z]?)$", t)
        if m: found.add(m.group(1))
    low = text.lower()
    for m in re.finditer(r"\b(?:e|ins)\s?-?(\d{3}[a-z]?)\b", low):
        found.add(m.group(1))
    for name, code in NAME_INDEX.items():
        if name in low: found.add(code)
    words = re.findall(r"[a-z]{6,}", low)  # fuzzy pass for single-word names
    singles = [n for n in NAME_INDEX if " " not in n and len(n) >= 6]
    for w in words:
        hit = difflib.get_close_matches(w, singles, n=1, cutoff=0.85)
        if hit: found.add(NAME_INDEX[hit[0]])
    return [{"code": "E" + c.upper(), "name": ADDITIVES[c][0], "risk": ADDITIVES[c][1]}
            for c in sorted(found) if c in ADDITIVES]


def score(n, additives, nova=None):
    """0-100 health score from per-100g nutrients, additives and NOVA group."""
    g = lambda k: float(n.get(k) or 0)
    kcal = g("energy-kcal_100g") or g("energy_100g") / 4.184
    s = 100 - min(30, g("sugars_100g") * 2) - min(20, g("saturated-fat_100g") * 2.5) \
        - min(20, g("salt_100g") * 15) - min(10, kcal / 60) \
        + min(10, g("fiber_100g") * 3) + min(10, g("proteins_100g"))
    s -= min(45, sum(PENALTY[a["risk"]] for a in additives))
    if nova == 4: s -= 10
    s = max(0, min(100, round(s)))
    label = "Excellent" if s >= 75 else "Good" if s >= 50 else "Poor" if s >= 25 else "Bad"
    return {"score": s, "label": label}


def decode_image(file):
    arr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Could not read image")
    return img


# ---------- Phase 1+2: barcode ----------
def read_barcode(img):
    det = cv2.barcode.BarcodeDetector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for im in (img, gray, cv2.resize(gray, None, fx=2, fy=2), cv2.resize(gray, None, fx=.5, fy=.5)):
        ok, codes, *_ = det.detectAndDecode(im)
        if ok:
            c = [x for x in codes if x]
            if c: return c[0]
    return None


def lookup(code):
    fields = "product_name,brands,image_front_small_url,nutriscore_grade,nova_group,nutriments,additives_tags,ingredients_text"
    r = requests.get(f"https://world.openfoodfacts.org/api/v2/product/{code}.json",
                     params={"fields": fields}, headers=UA, timeout=10)
    d = r.json()
    if d.get("status") != 1: return None
    p = d["product"]; n = p.get("nutriments", {})
    adds = find_additives(p.get("ingredients_text", ""), p.get("additives_tags", []))
    return {"source": "Open Food Facts", "name": p.get("product_name") or "Unknown product",
            "brand": p.get("brands"), "image": p.get("image_front_small_url"),
            "nutriscore": p.get("nutriscore_grade"), "nova": p.get("nova_group"),
            "nutrients": {k: n.get(k) for k in ("energy-kcal_100g", "sugars_100g", "saturated-fat_100g",
                          "salt_100g", "fiber_100g", "proteins_100g", "fat_100g", "carbohydrates_100g")},
            "ingredients": p.get("ingredients_text"), "additives": adds,
            **score(n, adds, p.get("nova_group"))}


@app.post("/api/barcode")
def api_barcode():
    code = request.form.get("code", "").strip()
    try:
        if not code and "image" in request.files:
            code = read_barcode(decode_image(request.files["image"]))
        if not code: return jsonify(error="No barcode found. Retake the photo closer, or type the number."), 422
        res = lookup(code)
        if not res: return jsonify(error=f"Barcode {code} is not in the database. Try the Label tab.", code=code), 404
        return jsonify(res | {"code": code})
    except Exception as e:
        return jsonify(error=str(e)), 500


# ---------- Phase 3: OCR ingredient label ----------
_reader = None
@app.post("/api/label")
def api_label():
    global _reader
    try:
        if _reader is None:
            import easyocr; _reader = easyocr.Reader(["en"], gpu=False)  # downloads model on first run
        img = decode_image(request.files["image"])
        text = " ".join(_reader.readtext(img, detail=0, paragraph=True))
        text = re.sub(r"\s+", " ", text)
        adds = find_additives(text)
        return jsonify(source="OCR", name="Scanned label", ingredients=text, additives=adds,
                       **score({}, adds, None) | {"note": "Score reflects additives only (no nutrition table read)."})
    except Exception as e:
        return jsonify(error=str(e)), 500


# ---------- Phase 4: food recognition via vision LLM ----------
@app.post("/api/food")
def api_food():
    try:
        import anthropic
        f = request.files["image"]
        b64 = base64.b64encode(f.read()).decode()
        prompt = ('Identify the food in the image. Reply with JSON only, no markdown: '
                  '{"name":str,"confidence":0-1,"serving_note":str,"nutrients":{"energy-kcal_100g":num,'
                  '"sugars_100g":num,"saturated-fat_100g":num,"salt_100g":num,"fiber_100g":num,'
                  '"proteins_100g":num,"fat_100g":num,"carbohydrates_100g":num}} using typical per-100g values.')
        msg = anthropic.Anthropic().messages.create(model=VISION_MODEL, max_tokens=600, messages=[{
            "role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": f.mimetype or "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt}]}])
        d = json.loads(re.sub(r"```(json)?", "", msg.content[0].text).strip())
        return jsonify(source="Vision AI (estimate)", additives=[], nova=None, **d, **score(d["nutrients"], []))
    except KeyError:
        return jsonify(error="No image uploaded."), 400
    except Exception as e:
        return jsonify(error=f"{e}  (Set ANTHROPIC_API_KEY to enable food recognition.)"), 500


@app.get("/")
def index(): return send_from_directory("static", "index.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
