from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
from frontendhelper import RecSys

app = Flask(__name__)
app.secret_key = "supersecretkey"  

# recommender
rs = RecSys()
rs.preprocess_data("abstract")
rs.build_vocab("abstract")
rs.compute_IDF("abstract")

rs.preprocess_data("title")
rs.build_vocab("title")
rs.compute_IDF("title")


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        qt = request.form.get("query_title")
        qa = request.form.get("query_abstract")

        scores = rs.similarity_ranking(qt, qa)
        top_idx = np.argsort(scores)[::-1][:5]

        results = []
        for idx in top_idx:
            results.append({
                "title": rs.dataset.iloc[idx]["title"],
                "abstract": rs.dataset.iloc[idx]["abstract"],
                "score": round(float(scores[idx]), 3)
            })

        session["results"] = results

        # refresh button
        return redirect(url_for("home"))

    # ✅ If GET request (after redirect)
    results = session.pop("results", None)  # get results once, then clear
    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
