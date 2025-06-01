from flask import Flask, request, render_template, flash, redirect, url_for
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import numpy as np
import io, os
from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
EMAIL_ADDRESS  = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
SMTP_SERVER    = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT      = int(os.getenv('SMTP_PORT', 465))

# Initialize Flask
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load YOLOv8 model
model = YOLO(r"D:\Palcode\VisioConstruct-AI\best.pt")
labels = ['door', 'window']

def pdf_bytes_to_images(pdf_bytes):
    """Convert a PDF in bytes to a list of PIL Image pages."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def preprocess_pil(img):
    """Resize PIL image to 640x640 for YOLO inference."""
    return img.resize((640, 640))

def detect_and_annotate(img):
    """
    Run YOLO on a PIL image, annotate it, and return (buffer, detections).
    """
    resized = preprocess_pil(img)
    results = model(resized)
    dets = []
    draw = ImageDraw.Draw(resized)
    font = ImageFont.load_default()
    colors = {'door': '#10B981', 'window': '#F97316'}

    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        if conf > 0.3:
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            label = labels[int(cls)]
            dets.append({'label': label, 'confidence': float(conf), 'bbox': [x, y, w, h]})
            c = colors[label]
            draw.rectangle([x, y, x + w, y + h], outline=c, width=3)
            draw.text((x, y - 10), f"{label} {conf:.2f}", fill=c, font=font)

    buf = io.BytesIO()
    resized.save(buf, format='PNG')
    buf.seek(0)
    return buf, dets

def create_combined_pdf(annotated_pages, total_counts, comment):
    """
    First page: summary. Then each annotated page.
    """
    pdf = FPDF()
    # Summary
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Blueprint Detection Report', ln=True, align='C')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 8, f"Total Doors: {total_counts['door']}", ln=True)
    pdf.cell(0, 8, f"Total Windows: {total_counts['window']}", ln=True)
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
    """
    Email the combined PDF report.
    """
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
        file    = request.files.get('file')
        email   = request.form.get('email')
        comment = request.form.get('comment', '')

        if not file or not email:
            flash('File and email are required', 'error')
            return redirect(request.url)

        data = file.read()
        total_counts     = {'door': 0, 'window': 0}
        annotated_pages  = []

        # Detect pages
        if file.filename.lower().endswith('.pdf'):
            pages = pdf_bytes_to_images(data)
        else:
            pages = [Image.open(io.BytesIO(data)).convert('RGB')]

        # Process each page
        for page in pages:
            buf, dets = detect_and_annotate(page)
            annotated_pages.append(buf)
            for d in dets:
                total_counts[d['label']] += 1

        # Save preview (first annotated page)
        preview_path = 'static/img/preview.png'
        with open(preview_path, 'wb') as f:
            f.write(annotated_pages[0].getvalue())

        # Build and send PDF
        pdf_buf = create_combined_pdf(annotated_pages, total_counts, comment)
        try:
            send_report(pdf_buf, email)
            flash('Report emailed successfully!', 'success')
        except Exception as e:
            flash(f'Email failed: {e}', 'error')

        return render_template(
            'index.html',
            result_image=url_for('static', filename='img/preview.png'),
            counts=total_counts
        )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
