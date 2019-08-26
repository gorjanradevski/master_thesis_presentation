from flask import render_template, url_for
from app import app
from app.forms import SearchForm
from app.inference import predict


@app.route("/", methods=["GET", "POST"])
def index():
    form = SearchForm()
    image_path = predict(form.query.data)
    return render_template(
        "index.html", title="SMHA Demo", images=image_path, form=form
    )
