from flask import Flask, render_template, request, url_for
import text_summary
from nltk import sent_tokenize

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyse", methods=["GET", "POST"])
def analyse():

    if request.method == "POST":
        text = request.form["text"]
    
        sentences = sent_tokenize(text)

        word_freq_dict = text_summary.sent_word_freq(text)

        tf_matrix = text_summary.tf(word_freq_dict)

        doc = text_summary.doc_frequency(word_freq_dict)

        idf_matrix = text_summary.idf(word_freq_dict, doc)

        tf_idf_matrix = text_summary.tf_idf(tf_matrix, idf_matrix)

        sent_score = text_summary.sentence_score(tf_idf_matrix)

        threshold = text_summary.average_score(sent_score)

        summary = text_summary.get_summary(sentences, sent_score, 1.1 * threshold)
        
    return render_template("summary.html", summary=summary)


@app.route("/back", methods=["GET", "POST"])
def back():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
