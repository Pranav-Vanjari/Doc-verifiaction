import os
from flask import Flask, render_template, request, jsonify, redirect, url_for

import main2  # Your ML model file

app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload_form():
    return render_template('upload_form.html')

# Validate File Before Submission (Real-Time)
@app.route('/validate_document', methods=['POST'])
def validate_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_type = request.form.get('file_type')  # Either 'Aadhar' or 'PAN'

    if file and allowed_file(file.filename):
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_" + file.filename)
        file.save(temp_path)

        # Predict the document type using ML model
        prediction = main2.predict(temp_path)
        os.remove(temp_path)  # Remove temp file after checking

        # Validate based on expected file type
        if (file_type == "Aadhar" and prediction == "Aadhar_Card") or \
           (file_type == "PAN" and prediction == "Pan_Card"):
            return jsonify({'valid': True, 'filename': file.filename})
        else:
            return jsonify({'valid': False, 'error': f"Invalid {file_type} document."})

    return jsonify({'error': 'Invalid file format'}), 400

# Save Files After Form Submission
@app.route('/upload_form', methods=['POST'])
def upload_page():
    if request.method == 'POST':
        name = request.form.get('name')
        aadhar = request.files.get('aadhar')
        pan = request.files.get('pan')

        if not aadhar or not pan:
            return render_template('upload_form.html', error="Both Aadhar and PAN card are required.")

        # Save Aadhar card
        if aadhar and allowed_file(aadhar.filename):
            aadhar_path = os.path.join(app.config['UPLOAD_FOLDER'], aadhar.filename)
            aadhar.save(aadhar_path)
        else:
            return render_template('upload_form.html', error="Invalid Aadhar card format.")

        # Save PAN card
        if pan and allowed_file(pan.filename):
            pan_path = os.path.join(app.config['UPLOAD_FOLDER'], pan.filename)
            pan.save(pan_path)
        else:
            return render_template('upload_form.html', error="Invalid PAN card format.")

        return redirect(url_for('success_page'))

@app.route('/success')
def success_page():
    return render_template('tick.html', message="Files successfully uploaded!")

if __name__ == '__main__':
    app.run(debug=True)
