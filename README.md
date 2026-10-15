# NutriScan (localhost)
```
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...                 # only needed for the Fresh food tab (Windows: set ...)
python app.py                                       # open http://localhost:5000
```
- Barcode: photo or typed number -> Open Food Facts -> score.
- Label: EasyOCR (downloads ~100 MB model on first use) -> additive matching.
- Fresh food: vision LLM estimate. Change model with VISION_MODEL env var.
- Edit the User-Agent email in app.py (Open Food Facts asks for it).
