from flask import Flask, render_template, request, redirect, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_bcrypt import Bcrypt
import numpy as np
import joblib
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "secret123"

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

# ================= USER =================
users = {
    "admin": bcrypt.generate_password_hash("admin123").decode('utf-8')
}

class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# ================= LOAD MODEL =================
model = joblib.load("model/fraud_model.pkl")

# ================= LOGIN =================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users and bcrypt.check_password_hash(users[username], password):
            login_user(User(username))
            return redirect("/dashboard")

    return render_template("login.html")

# ================= DASHBOARD =================
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

# ================= LOGOUT =================
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")

# ================= API =================
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json

    amount = float(data.get("amount", 0))
    time = float(data.get("time", 0))

    features = np.zeros(30)
    features[0] = time
    features[-1] = amount

    ml_pred = model.predict([features])[0]
    prob = model.predict_proba([features])[0][1]

    # Rule-based
    rule_pred = 1 if amount > 2000 else 0
    final_pred = 1 if rule_pred == 1 else ml_pred

    # Save for Power BI
    log_data(amount, time, final_pred, prob)

    return jsonify({
        "fraud": int(final_pred),
        "probability": float(prob)
    })

# ================= SAVE LOG =================
def log_data(amount, time, pred, prob):
    file = "output/live_data.csv"

    new_data = pd.DataFrame([{
        "Amount": amount,
        "Time": time,
        "Prediction": pred,
        "Probability": prob
    }])

    if not os.path.exists(file):
        new_data.to_csv(file, index=False)
    else:
        new_data.to_csv(file, mode='a', header=False, index=False)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)