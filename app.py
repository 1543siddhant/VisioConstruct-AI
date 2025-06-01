from flask import Flask, request, render_template, flash, redirect, url_for
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
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
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
SMTP_SERVER   = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT     = int(os.getenv('SMTP_PORT', 465))

# Initialize Flask
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load YOLOv8 model
model = YOLO(r"D:\Palcode\VisioConstruct-AI\best.pt")
labels = ['door', 'window']

# Helpers

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_resized = img.resize((640, 640))
    return img, img_resized

def annotate(img, detections):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    colors = {'door':'#10B981','window':'#F97316'}
    for d in detections:
        x,y,w,h = d['bbox']
        c = colors[d['label']]
        draw.rectangle([x,y,x+w,y+h], outline=c, width=3)
        draw.text((x,y-10), f"{d['label']} {d['confidence']:.2f}", fill=c, font=font)
    buf=io.BytesIO(); img.save(buf,format='PNG'); buf.seek(0)
    return buf

def create_pdf(image_buf, detections, comment):
    pdf=FPDF(); pdf.add_page()
    pdf.set_font('Helvetica','B',16)
    pdf.cell(0,10,'Blueprint Detection Report',ln=True,align='C'); pdf.ln(5)
    counts={'door':0,'window':0}
    for d in detections: counts[d['label']]+=1
    pdf.set_font('Helvetica','',12)
    pdf.cell(0,8,f"Doors: {counts['door']}",ln=True)
    pdf.cell(0,8,f"Windows: {counts['window']}",ln=True)
    pdf.ln(5)
    if comment:
        pdf.multi_cell(0,6,f"Comment: {comment}"); pdf.ln(5)
    pdf.image(image_buf, w=pdf.epw)
    out=io.BytesIO(); pdf.output(out); out.seek(0)
    return out

def send_report(pdf_buf, recipient):
    msg=MIMEMultipart()
    msg['Subject']='Your Blueprint Report'
    msg['From']=EMAIL_ADDRESS; msg['To']=recipient
    msg.attach(MIMEText('Find attached your blueprint detection report.','plain'))
    part=MIMEApplication(pdf_buf.read(),_subtype='pdf')
    part.add_header('Content-Disposition','attachment',filename='report.pdf')
    msg.attach(part)
    with smtplib.SMTP_SSL(SMTP_SERVER,SMTP_PORT) as smtp:
        smtp.login(EMAIL_ADDRESS,EMAIL_PASSWORD)
        smtp.send_message(msg)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='POST':
        if 'file' not in request.files:
            flash('Image required','error'); return redirect(request.url)
        file=request.files['file']
        email=request.form.get('email'); comment=request.form.get('comment','')
        if not file.filename or not email:
            flash('Image and email required','error'); return redirect(request.url)
        img_bytes=file.read()
        orig, resized = preprocess(img_bytes)
        results=model(resized)
        dets=[]
        for box in results[0].boxes.data.tolist():
            x1,y1,x2,y2,conf,cls=box
            if conf>0.3:
                dets.append({
                  'label':labels[int(cls)],
                  'confidence':float(conf),
                  'bbox':[int(x1),int(y1),int(x2-x1),int(y2-y1)]
                })
        ann_buf=annotate(resized.copy(),dets)
        pdf_buf=create_pdf(ann_buf.getvalue(),dets,comment)
        try:
            send_report(pdf_buf,email)
            flash('Report emailed successfully!','success')
        except Exception as e:
            flash(f'Email failed: {e}','error')
        # save preview
        with open('static/img/preview.png','wb') as f: f.write(ann_buf.getvalue())
        # compute totals
        counts={'door':0,'window':0}
        for d in dets: counts[d['label']]+=1
        return render_template('index.html',
                               result_image='static/img/preview.png',
                               counts=counts)
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True, use_reloader=False)
