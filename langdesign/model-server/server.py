from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/design', methods=['POST'])
def design():
    if request.method != 'POST':
        return jsonify({'error': 'Only POST requests are allowed'}), 400
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image']
    
    prompt = request.form.get('prompt')
    
    image_pil = Image.open(image)
    image_pil = image_pil.convert('RGB')
    transposed_image = image_pil.transpose(Image.TRANSPOSE)
    
    img_byte_array = io.BytesIO()
    transposed_image.save(img_byte_array, format='JPEG')

    #filepath = './da.jpeg'
    
    # Save the image
    #transposed_image.save(filepath, format='JPEG')
    
    img_byte_array.seek(0)
    return send_file(img_byte_array, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)