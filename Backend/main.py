from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
from flask_bcrypt import Bcrypt
from flask_pymongo import PyMongo
from dotenv import load_dotenv
from bson.objectid import ObjectId
import jwt
import datetime
import time
from eval_decoding import getEegResults
from bson import ObjectId, json_util
from flask import Response


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['MONGO_URI'] = os.getenv('MONGO_CONNECTION_STRING')
mongo = PyMongo(app)
bcrypt = Bcrypt(app)
CORS(app)  # Enable CORS for all routes

SECRET_KEY = os.getenv('JWT_SECRET')

users_collection = mongo.db.users
results_collection = mongo.db.uploaded_eegs

UPLOAD_FOLDER = './eeg'  # Set the path to your upload folder
ALLOWED_EXTENSIONS = {'mat'}  # Set the allowed file extensions
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_credentials():
    print("In credentials")
    data = request.get_json()
    print(data)
    print(type(data))
    if not isinstance(data, dict):
        print("Invalid JSON")
        abort(400, description="Invalid JSON data")
    username = data.get('username')
    password = data.get('password')
    print(username, password)
    if not username or not password:
        print("No password")
        abort(400, description="Username and password are required")
    return username, password

@app.route('/verify_token', methods=['POST'])
def verify_token():
    try:
        token = request.headers.get('Authorization')
        print(token)
        jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    except jwt.InvalidTokenError:
        abort(401, description="Invalid token")
    return jsonify({'message': 'Token verified successfully!!'}), 200


@app.route('/signup', methods=['POST'])
def signup():
    print("In signup")
    username, password = get_credentials()

    # Check if username already exists
    existing_user = users_collection.find_one({'username': username})
    if existing_user:
        abort(400, description="Username already exists")

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user_id = users_collection.insert_one({'username': username, 'password': hashed_password}).inserted_id
    user = users_collection.find_one({'username': username})
    #Generate session token
    token = jwt.encode({
        'user_id': str(user['_id']),
        'username':str(user['username']),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
    }, SECRET_KEY)
    print(token)
    print("User created")
    return jsonify({'message': 'User created successfully', 'token': token, 'username': username, 'user_id': str(user_id)}), 201

@app.route('/login', methods=['POST'])
def login():
    username, password = get_credentials()
    print(username,password)
    user = users_collection.find_one({'username': username})
    if not user or not bcrypt.check_password_hash(user['password'], password):
        abort(401, description="Invalid username or password")

    # Generate a session token
    token = jwt.encode({
        'user_id': str(user['_id']),
        'username':str(user['username']),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=300)
    }, SECRET_KEY)
    print(token)
    return jsonify({'message': 'Login successful', 'token': token, 'username': user['username'] }), 200


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_eeg', methods=['POST'])
def upload_eeg():
    auth_header = request.headers.get('Authorization')
    print(auth_header)
    if not auth_header or 'Bearer ' not in auth_header:
        abort(401, description="Missing or invalid Authorization header")
    token = auth_header.split(" ")[1]
    print("Token:", token)
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    except jwt.InvalidTokenError:
        abort(401, description="Invalid token")
    
    user_id = decoded_token['user_id']
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))

    print(user_id)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    print("Check 1")
    file = request.files.get('file')
    print(file)
    print("Check 2")
    if file and allowed_file(file.filename):
        time_of_upload = str(int(time.time() * 1000))
        name_of_file = time_of_upload + '.mat'
        filename = secure_filename(name_of_file)
        file_path = os.path.join(user_folder, filename)
        file.save(file_path)
        #return jsonify({'message': 'File uploaded successfully'}), 200
        predicted_para = getEegResults(str(os.path.join(user_folder, name_of_file)))
        result_document = {
            'user_id': ObjectId(user_id),  # Assuming you want to reference the user ID as an ObjectId
            # 'actual_para': actual_para,
            'predicted_para': predicted_para,
            # 'cos_sim': cos_sim,
            'file_name': name_of_file,
            'upload_time': datetime.datetime.utcnow()  # Store the current time of upload if needed
        }
        result_id = results_collection.insert_one(result_document).inserted_id
        return jsonify({
            # 'actual_para': actual_para,
            'predicted_para': predicted_para,
            # 'cos_sim': cos_sim
        })
    else:
        return jsonify({'error': 'Invalid file type. Only .mat files are allowed'}), 400
    

@app.route('/get_user_results', methods=['GET'])
def get_user_results():
    auth_header = request.headers.get('Authorization')
    if not auth_header or 'Bearer ' not in auth_header:
        abort(401, description="Missing or invalid Authorization header")

    token = auth_header.split(" ")[1]
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    except jwt.InvalidTokenError:
        abort(401, description="Invalid token")

    user_id_str = decoded_token['user_id']
    print(f"User ID string from token: {user_id_str}")

    user_id = ObjectId(user_id_str)
    user_results_cursor = results_collection.find({'user_id': user_id})
    
    results_list = list(user_results_cursor)
    results_json = json_util.dumps(results_list)

    return Response(results_json, mimetype='application/json')


@app.route('/delete_result/<result_id>', methods=['DELETE'])
def delete_result(result_id):
    auth_header = request.headers.get('Authorization')
    if not auth_header or 'Bearer ' not in auth_header:
        abort(401, description="Missing or invalid Authorization header")
    token = auth_header.split(" ")[1]
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    except jwt.InvalidTokenError:
        abort(401, description="Invalid token")
    
    user_id = ObjectId(decoded_token['user_id'])
    result_id = ObjectId(result_id)  # Convert the result_id from URL to ObjectId

    # Perform the deletion
    result = results_collection.delete_one({'_id': result_id, 'user_id': user_id})
    
    if result.deleted_count:
        return jsonify({'message': 'Result deleted successfully'}), 200
    else:
        return jsonify({'error': 'Result not found or permission denied'}), 404
    

@app.route('/get_result/<result_id>', methods=['GET'])
def get_result(result_id):
    auth_header = request.headers.get('Authorization')
    if not auth_header or 'Bearer ' not in auth_header:
        abort(401, description="Missing or invalid Authorization header")
    token = auth_header.split(" ")[1]
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    except jwt.InvalidTokenError:
        abort(401, description="Invalid token")
    
    result_id = ObjectId(result_id)  # Convert the result_id from URL to ObjectId

    # Fetch the document
    document = results_collection.find_one({'_id': result_id, 'user_id': ObjectId(decoded_token['user_id'])})
    
    if document:
        # Convert ObjectId to string for JSON serialization
        document['_id'] = str(document['_id'])
        document['user_id'] = str(document['user_id'])  # Assuming user_id is also an ObjectId
        return jsonify(document), 200
    else:
        return jsonify({'error': 'Document not found or permission denied'}), 404


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)