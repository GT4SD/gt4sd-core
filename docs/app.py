import os

import flask
import flask_login

app = flask.Flask(__name__, static_url_path="/", static_folder="_build/html/")
app.secret_key = "super secret string"  # Change this!

login_manager = flask_login.LoginManager()
login_manager.init_app(app)

users = {"gt4sd-team": {"password": os.environ["GT4SD_DOCS_PASSWORD"]}}


class User(flask_login.UserMixin):
    pass


@login_manager.user_loader
def user_loader(email):
    if email not in users:
        return

    user = User()
    user.id = email
    return user


@login_manager.request_loader
def request_loader(request):
    email = request.form.get("email")
    if email not in users:
        return

    user = User()
    user.id = email

    # DO NOT ever store passwords in plaintext and always compare password
    # hashes using constant-time comparison!
    user.is_authenticated = request.form["password"] == users[email]["password"]

    return user


@app.route("/login", methods=["GET", "POST"])
def login():
    if flask.request.method == "GET":
        return """
               <form action='login' method='POST'>
                <input type='text' name='email' id='email' placeholder='email'/>
                <input type='password' name='password' id='password' placeholder='password'/>
                <input type='submit' name='submit'/>
               </form>
               """

    email = flask.request.form["email"]
    if flask.request.form["password"] == users[email]["password"]:
        user = User()
        user.id = email
        flask_login.login_user(user)
        return flask.redirect(flask.url_for("serve_sphinx_docs"))

    return "Bad login"


@app.route("/logout")
def logout():
    flask_login.logout_user()
    return "Logged out"


@login_manager.unauthorized_handler
def unauthorized_handler():
    return flask.redirect(flask.url_for("login"))


@app.route("/")
@app.route("/<path:path>")
@flask_login.login_required
def serve_sphinx_docs(path="index.html"):
    return app.send_static_file(path)


if __name__ == "__main__":
    app.run(debug=True)
