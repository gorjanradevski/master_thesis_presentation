from flask import render_template, url_for
from app import app
from app.forms import SearchForm
from app.inference import predict


@app.route("/", methods=["GET", "POST"])
def index():
    form = SearchForm()
    if form.query.data is not None:
        image_paths = predict(form.query.data)
        return render_template("index.html", image_path=image_paths, form=form)
    return render_template("index.html", form=form)
