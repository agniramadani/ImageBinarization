import os
import subprocess
import zipfile
import io
import shutil

from flask import Flask, send_from_directory, request, redirect, url_for, make_response

app = Flask(__name__)

UPLOAD_FOLDER = os.path.abspath("./uploaded_images")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def serve_static_index():
    return send_from_directory('templates', 'index.html')


@app.route('/logo.png')
def serve_static_image():
    return send_from_directory('templates', 'logo.png')


@app.route('/upload', methods=['POST'])
def upload():
    # Before we upload content of directories must be empty
    directories_to_clear = ['binarized_images', 'uploaded_images']
    for directory_path in directories_to_clear:
        # Remove the directory and its contents
        shutil.rmtree(directory_path)
        # Recreate the empty directory
        os.makedirs(directory_path)

    # Get the list of files uploaded by the user
    files = request.files.getlist('file')
    # Check if any files were uploaded
    if not files:
        return "No files provided"

    # Save each uploaded file to the UPLOAD_FOLDER directory
    for file in files:
        filename = file.filename
        if filename == "":
            return "No file selected"
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

    # Redirect the user to the binarize route
    return redirect(url_for("binarize"))


@app.route('/binarizing')
def binarize():
    # Execute the test.py script as a subprocess
    p = subprocess.Popen(['python', 'test.py'])
    # Wait for the child process to finish
    p.wait()
    # Redirect the user when subprocess is done
    return redirect(url_for("done"))


@app.route('/done')
def done():
    # Serve the static success.html file
    return send_from_directory('templates', 'success.html')


@app.route('/download', methods=['POST'])
def download():
    # Replace 'path/to/directory' with the path to your directory
    # This is the directory that contains the binarized images that will be zipped and downloaded
    directory_path = 'binarized_images'

    # Create a memory buffer to hold the zip file contents
    buffer = io.BytesIO()

    # Create a zip file object to write the contents of the directory to
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Iterate over all the files in the directory and add them to the zip file
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, directory_path))

    # Reset the buffer position to the start
    buffer.seek(0)

    # Create a Flask response object from the buffer contents
    response = make_response(buffer.getvalue())

    # Set the response headers to indicate that this is a zip file
    response.headers['Content-Type'] = 'application/zip'
    response.headers['Content-Disposition'] = 'attachment; filename=directory.zip'

    return response


if __name__ == '__main__':
    app.run()
