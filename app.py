from flask import Flask, request, jsonify
import os
import sys
import psycopg2
# from pgvector.psycopg2 import register_vector
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'PaddleOCR2Pytorch')))
import cv2
import time
from PIL import Image
from PaddleOCR2Pytorch.pytorchocr.utils.utility import get_image_file_list_request
from PaddleOCR2Pytorch.tools.infer.pytorchocr_utility import draw_ocr_box_txt, parse_args
from PaddleOCR2Pytorch.tools.infer.predict_system import TextSystem
import openai
from util import retrieve_nearest_embedding,get_ada_embedding
import uuid
app = Flask(__name__)

# Sử dụng biến môi trường để thiết lập kết nối
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'postgres')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
OpenAIKey = os.getenv('OpenAIKey', None)
rec_image_shape = os.getenv('rec_image_shape', None)
det_yaml_path = os.getenv('det_yaml_path', None)
rec_yaml_path = os.getenv('rec_yaml_path', None)
rec_char_dict_path = os.getenv('rec_char_dict_path', None)
rec_model_path = os.getenv('rec_model_path', None)
det_model_path = os.getenv('det_model_path', None)
cls_model_path = os.getenv('cls_model_path', None)

def generate_uuid_v4():
    return str(uuid.uuid4())

@app.route('/OCR', methods=['POST'])
def OCR():
    response = request.get_json()
    url = response['url']
    args = parse_args()
    args.rec_image_shape = rec_image_shape
    args.det_yaml_path = det_yaml_path
    args.rec_yaml_path = rec_yaml_path
    args.rec_char_dict_path = rec_char_dict_path
    # path model converter
    args.rec_model_path = rec_model_path
    args.det_model_path = det_model_path
    args.cls_model_path = cls_model_path
    ###
    embedding_text = []
    image_file_list = get_image_file_list_request(url)
    text_sys = TextSystem(args)
    for img in image_file_list:
        dt_boxes, rec_res = text_sys(img)
        result_string = " ".join(text for text, _ in rec_res)
        embedding = get_ada_embedding(result_string,OpenAIKey)
        embedding_text.append({"embedding":embedding, "text": result_string})

    conn = psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT
    )
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            print("Extension 'vector' đã tồn tại.")
        else:
            # Nếu extension "vector" chưa tồn tại, thêm nó vào cơ sở dữ liệu
            cur.execute("CREATE EXTENSION vector;")
            print("Extension 'vector' đã được thêm vào cơ sở dữ liệu.")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id UUID PRIMARY KEY,
            embedding VECTOR(1536),  
            content TEXT 
        );
        """)
        for v in embedding_text:
            _id = generate_uuid_v4()
            cur.execute("INSERT INTO items (id,embedding, content) VALUES (%s, %s, %s);", (_id, v["embedding"], v["text"]))
    conn.commit()
    conn.close()

    return jsonify({"message": "Request processed successfully "})


@app.route('/chat', methods=['POST'])
def generate_response():
    openai.api_key = OpenAIKey
    data = request.get_json()
    if 'input' not in data:
        return jsonify({'error': 'No input provided'}), 400
    user_input = data['input']

    # Tạo embedding cho truy vấn của người dùng
    query_embedding_response = openai.Embedding.create(
        input=user_input,
        engine="text-embedding-ada-002"
    )
    query_embedding = query_embedding_response['data'][0]['embedding']
    # Truy xuất văn bản gần nhất từ PostgreSQL
    conn = psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT
    )
    nearest_item = retrieve_nearest_embedding(query_embedding,conn)
    chat = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Based on the nearest content: '{nearest_item[1]}', {user_input}"}
        ]
    )
    reply = chat.choices[0].message.content

    return jsonify({'response': reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8868)
