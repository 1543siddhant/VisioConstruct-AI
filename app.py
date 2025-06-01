import os
import io
import smtplib

from flask import Flask, request, render_template, flash, redirect, url_for
from fpdf import FPDF
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from dotenv import load_dotenv

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import fitz  # PyMuPDF
import onnxruntime as ort

# Load environment variables
load_dotenv()
EMAIL_ADDRESS  = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
SMTP_SERVER    = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT      = int(os.getenv('SMTP_PORT', 465))

# Initialize Flask
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load ONNX model
ONNX_PATH = os.path.join(os.getcwd(), "best.onnx")
if not os.path.exists(ONNX_PATH):
    raise FileNotFoundError(f"ONNX model not found at {ONNX_PATH}")
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

labels = ['door', 'window']

def pdf_bytes_to_images(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    return pages

def detect_and_annotate(img: Image.Image):
    # resize to 640×640
    resized = img.resize((640, 640))
    arr = np.array(resized, dtype=np.float32).transpose(2,0,1)[None] / 255.0

    # run ONNX
    outs = session.run(None, {input_name: arr})[0][0]  # shape [N,6]

    dets = []
    draw = ImageDraw.Draw(resized)
    font = ImageFont.load_default()
    colors = {'door': '#10B981', 'window': '#F97316'}

    for x1, y1, x2, y2, conf, cls in outs:
        if conf > 0.3:
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            label = labels[int(cls)]
            dets.append({'label': label, 'confidence': float(conf), 'bbox': [x, y, w, h]})
            c = colors[label]
            draw.rectangle([x, y, x+w, y+h], outline=c, width=3)
            draw.text((x, y-10), f"{label} {conf:.2f}", fill=c, font=font)

    buf = io.BytesIO()
    resized.save(buf, format='PNG')
    buf.seek(0)
    return buf, dets

def create_combined_pdf(annotated_pages, counts, comment):
    pdf = FPDF()
    # Summary page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Blueprint Detection Report', ln=True, align='C')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 8, f"Total Doors: {counts['door']}", ln=True)
    pdf.cell(0, 8, f"Total Windows: {counts['window']}", ln=True)
    pdf.ln(5)
    if comment:
        pdf.multi_cell(0, 6, f"Comment: {comment}")
        pdf.ln(5)

    # Annotated pages
    for buf in annotated_pages:
        pdf.add_page()
        pdf.image(buf, w=pdf.epw)

    out = io.BytesIO()
    pdf.output(out)
    out.seek(0)
    return out

def send_report(pdf_buf, recipient):
    msg = MIMEMultipart()
    msg['Subject'] = 'Your Blueprint Report'
    msg['From']    = EMAIL_ADDRESS
    msg['To']      = recipient
    msg.attach(MIMEText('Please find attached your detection report.', 'plain'))

    part = MIMEApplication(pdf_buf.read(), _subtype='pdf')
    part.add_header('Content-Disposition', 'attachment', filename='report.pdf')
    msg.attach(part)

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f       = request.files.get('file')
        email   = request.form.get('email')
        comment = request.form.get('comment', '')

        if not f or not email:
            flash('File and email required', 'error')
            return redirect(request.url)

        data = f.read()
        # convert PDF or single image
        if f.filename.lower().endswith('.pdf'):
            pages = pdf_bytes_to_images(data)
        else:
            pages = [Image.open(io.BytesIO(data)).convert('RGB')]

        annotated_pages = []
        counts          = {'door': 0, 'window': 0}

        for p in pages:
            buf, dets = detect_and_annotate(p)
            annotated_pages.append(buf)
            for d in dets:
                counts[d['label']] += 1

        # preview first page
        preview = 'static/img/preview.png'
        with open(preview, 'wb') as out:
            out.write(annotated_pages[0].getvalue())

        report = create_combined_pdf(annotated_pages, counts, comment)
        try:
            send_report(report, email)
            flash('Report emailed!', 'success')
        except Exception as e:
            flash(f'Email failed: {e}', 'error')

        return render_template('index.html',
                               result_image=url_for('static', filename='img/preview.png'),
                               counts=counts)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
