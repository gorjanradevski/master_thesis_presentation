from flask import render_template, url_for
from app import app
from app.forms import SearchForm


@app.route("/", methods=["GET", "POST"])
def index():
    form = SearchForm()
    return render_template("index.html", title="SMHA Demo", form=form)
