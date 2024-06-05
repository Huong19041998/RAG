from flask import Flask, request, jsonify
import os
import queue
import openai
from util import retrieve_nearest_embedding,process_ocr_and_store, generate_uuid_v4,redis_client
import threading
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)

job_queue = queue.Queue()
num_worker_threads = 4
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'postgres')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
OpenAIKey = os.getenv('OpenAIKey', None)
def worker():
    while True:
        url, job_id = job_queue.get()
        try:
            process_ocr_and_store(url, job_id)
        finally:
            job_queue.task_done()
@app.route('/OCR', methods=['POST'])
def OCR():
    response = request.get_json()
    url = response['url']
    job_id = generate_uuid_v4()
    # threading.Thread(target=process_ocr_and_store, args=(url,job_id)).start()
    redis_client.set(job_id, 'queued')
    job_queue.put((url, job_id))
    return jsonify({"message": f"Request received, processing will continue in background ID: {job_id}"}), 200


@app.route('/chat', methods=['POST'])
def generate_response():
    openai.api_key = OpenAIKey
    data = request.get_json()
    if 'input' not in data:
        return jsonify({'error': 'No input provided'}), 400
    user_input = data['input']

    query_embedding_response = openai.Embedding.create(
        input=user_input,
        engine="text-embedding-ada-002"
    )
    query_embedding = query_embedding_response['data'][0]['embedding']
    nearest_item = retrieve_nearest_embedding(query_embedding)
    chat = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Based on the nearest content: '{nearest_item[1]}', {user_input}"}
        ]
    )
    reply = chat.choices[0].message.content

    return jsonify({'response': reply})
@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    status = redis_client.get(job_id)
    if not status:
        return jsonify({"error": "Invalid job ID"}), 404
    return jsonify({"job_id": job_id, "status": status}), 200
if __name__ == '__main__':
    for i in range(num_worker_threads):
        threading.Thread(target=worker, daemon=True).start()
    app.run(host='0.0.0.0', port=8868)
