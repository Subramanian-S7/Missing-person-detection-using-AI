import os
import time
import sqlite3
from datetime import datetime

from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, jsonify, send_file
)
from werkzeug.utils import secure_filename

import cv2
import numpy as np

import base64
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


from twilio.rest import Client

TWILIO_SID = "ACe8341336be8357a40bc265daf97d03d6"
TWILIO_AUTH = "9e5b22a7da9e11245ee610d74535be48"
TWILIO_WHATSAPP = "whatsapp:+14155238886"   # Sandbox number

client = Client(TWILIO_SID, TWILIO_AUTH)

def send_whatsapp_alert(to_number, message):
    client.messages.create(
        body=message,
        from_="whatsapp:+14155238886",
        to="whatsapp:" + to_number
)



# ---------- Paths & Flask setup ----------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "missing_persons.db")

app = Flask(__name__)
app.secret_key = "change-this-for-project"

# static folder already known by Flask
IMAGE_SUBDIR = "person_images"
IMAGE_DIR = os.path.join(app.static_folder, IMAGE_SUBDIR)
os.makedirs(IMAGE_DIR, exist_ok=True)


# ---------- DB helpers ----------

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            father_name TEXT,
            address TEXT,
            email TEXT NOT NULL,
            phone TEXT,
            aadhar TEXT,
            dob TEXT,
            missing_date TEXT,
            image_path TEXT,
            is_found INTEGER DEFAULT 0,
            last_seen_time TEXT,
            last_seen_location TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def insert_case(data):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO cases (
            first_name, last_name, father_name, address,
            email, phone, aadhar, dob, missing_date,
            image_path, is_found
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (
            data["first_name"], data["last_name"], data["father_name"],
            data["address"], data["email"], data["phone"], data["aadhar"],
            data["dob"], data["missing_date"], data["image_path"]
        ),
    )
    conn.commit()
    conn.close()


def get_stats():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM cases WHERE is_found = 0")
    active = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM cases WHERE is_found = 1")
    reunited = cur.fetchone()[0]
    conn.close()
    return active, reunited


def get_case(case_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM cases WHERE id = ?", (case_id,))
    row = cur.fetchone()
    conn.close()
    return row


def get_open_cases():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM cases WHERE is_found = 0")
    rows = cur.fetchall()
    conn.close()
    return rows

def get_all_cases():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM cases ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_found_cases():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM cases WHERE is_found = 1 ORDER BY last_seen_time DESC")
    rows = cur.fetchall()
    conn.close()
    return rows



def mark_case_found(case_id, detection_time, location):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE cases
        SET is_found = 1,
            last_seen_time = ?,
            last_seen_location = ?
        WHERE id = ?
        """,
        (detection_time, location, case_id),
    )
    conn.commit()
    conn.close()


# ---------- Face recognition helpers (OpenCV LBPH) ----------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def load_training_data():
    """
    Load face images from static/person_images and train LBPH recognizer.
    Only for cases where is_found = 0.
    """
    cases = get_open_cases()
    faces = []
    labels = []

    for case in cases:
        rel_path = case["image_path"]  # e.g. "person_images/123.jpg"
        full_path = os.path.join(app.static_folder, rel_path)

        if not os.path.exists(full_path):
            continue

        img = cv2.imread(full_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5
        )

        for (x, y, w, h) in detected:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (200, 200))
            faces.append(face_roi)
            labels.append(case["id"])
            break  # one face per image is enough

    if not faces:
        print("[WARN] No faces found in training data.")
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    print(f"[INFO] Trained recognizer on {len(labels)} face(s).")
    return recognizer


def format_phone_number(raw_phone: str | None) -> str | None:
    """
    Normalize a phone number to E.164 for WhatsApp / Twilio.
    Examples in  -> out:
      '9876543210'      -> '+919876543210'
      '919876543210'    -> '+919876543210'
      '+919876543210'   -> '+919876543210'
    """
    if not raw_phone:
        return None

    phone = str(raw_phone).strip().replace(" ", "").replace("-", "")

    # Already in +<country><number> format
    if phone.startswith("+"):
        return phone

    # Starts with country code but missing '+'
    if phone.startswith("91") and len(phone) == 12:
        return "+" + phone

    # Plain 10-digit Indian mobile
    if len(phone) == 10 and phone.isdigit():
        return "+91" + phone

    # Fallback: assume India and add +91
    return "+91" + phone


def run_surveillance(video_source=0, confidence_threshold=70):
    """
    Opens the webcam in a separate OpenCV window,
    runs face recognition until:
      - a match is found, or
      - user presses 'q'.

    Returns (case_id, detection_time, location) or (None, None, None).
    """
    recognizer = load_training_data()
    if recognizer is None:
        return None, None, None

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return None, None, None

    print("[INFO] Surveillance started. Press 'q' to quit.")

    matched_case_id = None
    detection_time = None
    location = "Camera 1 – Main Gate"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            label, confidence = recognizer.predict(face_roi)

            if confidence < confidence_threshold:
                matched_case_id = int(label)
                detection_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Match ID {label} ({confidence:.1f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                print(
                    f"[MATCH] Case ID {label} at {detection_time} "
                    f"(conf={confidence:.2f})"
                )

                # Show frame for a moment so user can see the match box
                cv2.imshow("Surveillance - Press q to quit", frame)
                cv2.waitKey(1000)
                break  # break faces loop
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cv2.imshow("Surveillance - Press q to quit", frame)

        # If we found a match, break out of main loop
        if matched_case_id is not None:
            break

        # Allow user to exit manually
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup camera and window
    cap.release()
    cv2.destroyAllWindows()

    # If a match was detected, mark found + send WhatsApp
    if matched_case_id is not None:
        # 1) Update DB
        mark_case_found(matched_case_id, detection_time, location)

        # 2) Fetch case and phone
        case = get_case(matched_case_id)
        if case:
            # sqlite3.Row behaves like a dict but without .get()
            phone_value = None
            keys = case.keys()

            if "phone" in keys:
                phone_value = case["phone"]
            elif "phone_number" in keys:
                phone_value = case["phone_number"]

            to_number = format_phone_number(phone_value)

            if to_number:
                message = (
                    f"🚨 Missing Person Found!\n\n"
                    f"Case ID: {matched_case_id}\n"
                    f"Name: {case['first_name']} {case['last_name']}\n"
                    f"Detected at: {detection_time}\n"
                    f"Location: {location}\n\n"
                    f"- Bharatiya Rescue System"
                )

                # Your helper: def send_whatsapp_alert(to_number, message)
                send_whatsapp_alert(to_number, message)
                print(f"[INFO] WhatsApp alert sent to {to_number}")
            else:
                print("[WARN] No valid phone number; WhatsApp not sent.")
        else:
            print("[WARN] Case not found in DB; WhatsApp not sent.")

        return matched_case_id, detection_time, location

    # No match case
    return None, None, None





# ---------- Routes ----------

@app.route("/")
def home():
    active, reunited = get_stats()
    return render_template(
        "home.html",
        stat_active=active,
        stat_reunited=reunited
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        first_name = request.form.get("first_name", "").strip()
        last_name = request.form.get("last_name", "").strip()
        father_name = request.form.get("father_name", "").strip()
        address = request.form.get("address", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        aadhar = request.form.get("aadhar", "").strip()
        dob = request.form.get("dob", "").strip()
        missing_date = request.form.get("missing_date", "").strip()

        photo = request.files.get("photo")

        if not first_name or not last_name or not email:
            flash("First name, last name, and email are required.", "danger")
            return redirect(url_for("register"))

        if not photo or photo.filename == "":
            flash("Please upload a clear face image.", "danger")
            return redirect(url_for("register"))

        # save image into static/person_images
        filename = f"{int(time.time())}_{secure_filename(photo.filename)}"
        save_path = os.path.join(IMAGE_DIR, filename)
        photo.save(save_path)

        # store only relative path in DB (e.g. "person_images/123.jpg")
        image_rel_path = os.path.join(IMAGE_SUBDIR, filename).replace("\\", "/")

        insert_case(
            {
                "first_name": first_name,
                "last_name": last_name,
                "father_name": father_name,
                "address": address,
                "email": email,
                "phone": phone,
                "aadhar": aadhar,
                "dob": dob,
                "missing_date": missing_date,
                "image_path": image_rel_path,
            }
        )

        flash("Case registered successfully. You can now start surveillance.",
              "success")
        return redirect(url_for("surveillance"))

    return render_template("register.html")


@app.route("/surveillance", methods=["GET", "POST"])
def surveillance():
    if request.method == "POST":
        case_id, detection_time, location = run_surveillance()
        if case_id:
            return redirect(url_for("match_page", case_id=case_id))
        else:
            flash("No match detected. Try again.", "warning")
            return redirect(url_for("surveillance"))

    return render_template("surveillance.html")

@app.route("/cases")
def cases_list():
    cases = get_all_cases()
    return render_template("cases.html", cases=cases, view="all")


@app.route("/cases/found")
def found_cases():
    cases = get_found_cases()
    return render_template("cases.html", cases=cases, view="found")

@app.route("/case/<int:case_id>/edit", methods=["GET", "POST"])
def edit_case(case_id):
    case = get_case(case_id)
    if not case:
        flash("Case not found.", "danger")
        return redirect(url_for("cases_list"))

    if request.method == "POST":
        first_name = request.form.get("first_name", "").strip()
        last_name = request.form.get("last_name", "").strip()
        father_name = request.form.get("father_name", "").strip()
        address = request.form.get("address", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        aadhar = request.form.get("aadhar", "").strip()
        dob = request.form.get("dob", "").strip()
        missing_date = request.form.get("missing_date", "").strip()

        if not first_name or not last_name or not email:
            flash("First name, last name, and email are required.", "danger")
            return redirect(url_for("edit_case", case_id=case_id))

        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE cases
            SET first_name = ?, last_name = ?, father_name = ?, address = ?,
                email = ?, phone = ?, aadhar = ?, dob = ?, missing_date = ?
            WHERE id = ?
            """,
            (
                first_name, last_name, father_name, address,
                email, phone, aadhar, dob, missing_date, case_id
            ),
        )
        conn.commit()
        conn.close()

        flash("Case updated successfully.", "success")
        return redirect(url_for("cases_list"))

    # GET
    return render_template("edit_case.html", case=case)


@app.route("/case/<int:case_id>/delete", methods=["POST"])
def delete_case(case_id):
    case = get_case(case_id)
    if not case:
        flash("Case not found.", "danger")
        return redirect(url_for("cases_list"))

    # delete image file if exists
    if case["image_path"]:
        img_path = os.path.join(app.static_folder, case["image_path"])
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
            except OSError:
                pass

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM cases WHERE id = ?", (case_id,))
    conn.commit()
    conn.close()

    flash("Case deleted successfully.", "success")
    return redirect(url_for("cases_list"))


@app.route("/case/<int:case_id>/report")
def case_report(case_id):
    case = get_case(case_id)
    if not case:
        return "Case not found", 404

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50

    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, y, "Missing Person Report")
    y -= 30

    p.setFont("Helvetica", 11)
    p.drawString(50, y, f"Case ID: {case['id']}")
    y -= 18
    p.drawString(50, y, f"Name: {case['first_name']} {case['last_name']}")
    y -= 18
    p.drawString(50, y, f"Father's Name: {case['father_name'] or '-'}")
    y -= 18
    p.drawString(50, y, f"Address: {case['address'] or '-'}")
    y -= 18
    p.drawString(50, y, f"Email: {case['email']}")
    y -= 18
    p.drawString(50, y, f"Phone: {case['phone'] or '-'}")
    y -= 18
    p.drawString(50, y, f"Aadhar: {case['aadhar'] or '-'}")
    y -= 18
    p.drawString(50, y, f"Date of Birth: {case['dob'] or '-'}")
    y -= 18
    p.drawString(50, y, f"Missing Since: {case['missing_date'] or '-'}")
    y -= 24

    status_text = "Found" if case["is_found"] else "Not Found"
    p.drawString(50, y, f"Status: {status_text}")
    y -= 18
    p.drawString(50, y, f"Last Seen Time: {case['last_seen_time'] or '-'}")
    y -= 18
    p.drawString(50, y, f"Last Seen Location: {case['last_seen_location'] or '-'}")
    y -= 30

    p.setFont("Helvetica-Oblique", 10)
    p.drawString(50, y, "Generated by Bharatiya Rescue – Missing Person Detection System")

    p.showPage()
    p.save()

    buffer.seek(0)
    filename = f"case_{case_id}_report.pdf"
    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf",
    )



@app.route("/case/<int:case_id>")
def match_page(case_id):
    case = get_case(case_id)
    if not case:
        return "Case not found", 404
    return render_template("match.html", case=case)


if __name__ == "__main__":
    init_db()
    
    send_whatsapp_alert("+to number", "Test message from Missing Person System")

    app.run(debug=True)
