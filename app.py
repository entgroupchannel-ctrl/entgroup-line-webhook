import os, json
import pandas as pd
import numpy as np
from flask import Flask, request, Response
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

def safe_json(data):
    class Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)
    return Response(json.dumps(data, cls=Enc, ensure_ascii=False),
                    mimetype='application/json')

@app.after_request
def add_cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return r

@app.route("/", methods=["GET"])
def index():
    return safe_json({
        "status": "ok",
        "service": "LINE Sentiment Analyzer — ENT GROUP",
        "endpoints": ["POST /api/sentiment/analyze"]
    })

@app.route("/health", methods=["GET"])
def health():
    return safe_json({"status": "ok"})

@app.route("/api/sentiment/analyze", methods=["POST", "OPTIONS"])
def sentiment_analyze():
    if request.method == "OPTIONS":
        return safe_json({}), 200
    try:
        import time
        t0 = time.time()
        body = request.get_json()
        data = body.get("data", [])

        if not data:
            return safe_json({"error": "ไม่มีข้อมูล"}), 400

        df = pd.DataFrame(data)
        if "review_text" not in df.columns:
            return safe_json({"error": "Missing column: review_text"}), 400

        df["review_text"] = df["review_text"].fillna("").astype(str)
        has_label = ("sentiment" in df.columns and
                     df["sentiment"].notna().sum() > 0)

        X = df["review_text"]

        # ── Rule-based pseudo labels (ถ้าไม่มี label จริง) ──
        pos_w = ["ดี","เยี่ยม","ประทับใจ","แนะนำ","คุ้ม","ชอบ","ขอบคุณ",
                 "excellent","great","perfect","love","good","satisfied","thanks"]
        neg_w = ["แย่","ผิดหวัง","ห่วย","ช้า","เสีย","ไม่ดี","โกรธ","ไม่พอใจ",
                 "terrible","worst","poor","bad","horrible","disappointed","angry"]

        def rule_label(text):
            tl = text.lower()
            p = sum(1 for w in pos_w if w in tl)
            n = sum(1 for w in neg_w if w in tl)
            return "Positive" if p > n else "Negative" if n > p else "Neutral"

        y = df["sentiment"].fillna("Neutral") if has_label else X.apply(rule_label)

        # ── Train Logistic + TF-IDF ──
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=1500, ngram_range=(1,2),
                                      sublinear_tf=True, min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, C=0.5, random_state=42))
        ])
        pipe.fit(X, y)

        # ── Predict all ──
        preds     = pipe.predict(X)
        proba_all = pipe.predict_proba(X)
        classes   = pipe.classes_

        # ── Keywords per class ──
        tfidf_step = pipe.named_steps["tfidf"]
        feat_names = tfidf_step.get_feature_names_out()
        keywords = {}
        for cls in ["Positive", "Negative", "Neutral"]:
            mask = y == cls
            if mask.sum() > 0:
                vecs = tfidf_step.transform(X[mask])
                sc   = vecs.mean(axis=0).A1
                top  = sc.argsort()[-8:][::-1]
                keywords[cls] = [str(feat_names[i]) for i in top
                                 if len(str(feat_names[i])) > 1][:6]

        # ── Build predictions list ──
        predictions = []
        for i, row in df.iterrows():
            prob = float(proba_all[i].max())
            pred = preds[i]
            prob_dict = {c: round(float(p)*100, 1)
                         for c, p in zip(classes, proba_all[i])}
            predictions.append({
                "review_id":          str(row.get("review_id", f"MSG{i:04d}")),
                "review_text":        str(row.get("review_text",""))[:300],
                "channel":            str(row.get("channel", "")),
                "date":               str(row.get("date", "")),
                "month":              str(row.get("month", "")),
                "display_name":       str(row.get("display_name", "")),
                "topic":              str(row.get("topic", "")),
                "actual_sentiment":   str(row.get("sentiment","")) if has_label else None,
                "predicted_sentiment": pred,
                "confidence":         round(prob, 3),
                "probabilities":      prob_dict,
            })

        # ── Summary ──
        cnt   = Counter(preds)
        total = len(predictions)
        pos   = int(cnt.get("Positive", 0))
        neg   = int(cnt.get("Negative", 0))
        neu   = int(cnt.get("Neutral", 0))

        # ── Channel breakdown ──
        channel_breakdown = {}
        if "channel" in df.columns:
            for ch in df["channel"].dropna().unique():
                mask = df["channel"] == ch
                cp   = Counter(p["predicted_sentiment"]
                               for p, m in zip(predictions, mask) if m)
                n    = sum(cp.values())
                channel_breakdown[str(ch)] = {
                    "total":        n,
                    "Positive":     int(cp.get("Positive", 0)),
                    "Negative":     int(cp.get("Negative", 0)),
                    "Neutral":      int(cp.get("Neutral", 0)),
                    "negative_pct": round(cp.get("Negative",0)/n*100, 1) if n else 0,
                }

        # ── Topic breakdown ──
        topic_breakdown = {}
        if "topic" in df.columns:
            for tp in df["topic"].dropna().unique():
                mask = df["topic"] == tp
                tp_c = Counter(p["predicted_sentiment"]
                               for p, m in zip(predictions, mask) if m)
                topic_breakdown[str(tp)] = {
                    "Positive": int(tp_c.get("Positive", 0)),
                    "Negative": int(tp_c.get("Negative", 0)),
                    "Neutral":  int(tp_c.get("Neutral", 0)),
                }

        # ── Monthly trend ──
        monthly_trend = {}
        month_col = "month" if "month" in df.columns else None
        if month_col:
            for mo in sorted(df[month_col].dropna().unique()):
                mask = df[month_col] == mo
                mc   = Counter(p["predicted_sentiment"]
                               for p, m in zip(predictions, mask) if m)
                monthly_trend[str(mo)] = {
                    "Positive": int(mc.get("Positive", 0)),
                    "Negative": int(mc.get("Negative", 0)),
                    "Neutral":  int(mc.get("Neutral", 0)),
                }

        elapsed = round(time.time() - t0, 1)
        print(f"✅ sentiment: {total}msgs | +{pos} -{neg} ~{neu} | {elapsed}s")

        return safe_json({
            "total_messages":   total,
            "processing_time":  elapsed,
            "has_label":        has_label,
            "sentiment_summary": {
                "Positive":      pos,
                "Negative":      neg,
                "Neutral":       neu,
                "positive_pct":  round(pos/total*100, 1) if total else 0,
                "negative_pct":  round(neg/total*100, 1) if total else 0,
                "neutral_pct":   round(neu/total*100, 1) if total else 0,
                "net_score":     round((pos-neg)/total*100, 1) if total else 0,
            },
            "channel_breakdown": channel_breakdown,
            "topic_breakdown":   topic_breakdown,
            "monthly_trend":     monthly_trend,
            "keywords":          keywords,
            "predictions":       predictions,
            "top_negative": sorted(
                [p for p in predictions if p["predicted_sentiment"]=="Negative"],
                key=lambda x: x["confidence"], reverse=True)[:10],
        })

    except Exception as e:
        import traceback; print(traceback.format_exc())
        return safe_json({"error": str(e)}), 500
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
