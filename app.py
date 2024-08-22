from flask import Flask, request, jsonify
import os
import openai
from util import retrieve_nearest_embedding,process_ocr_and_store, generate_uuid_v4,redis_client,queue_rd,run_worker
from concurrent.futures import ThreadPoolExecutor
# import subprocess
from multiprocessing import Process
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

num_workers = 4
OpenAIKey = os.getenv('OpenAIKey', None)


@app.route('/OCR', methods=['POST'])
def OCR():
    response = request.get_json()
    print("response",response['url'])
    url = response['url']
    job_id = generate_uuid_v4()
    redis_client.set(job_id, 'queued')
    redis_client.setex(job_id, 600 , 'queued') #timedelta(days=1)
    queue_rd.enqueue(process_ocr_and_store, url, job_id)
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
    status = status.decode('utf-8')
    return jsonify({"job_id": job_id, "status": status}), 200

if __name__ == '__main__':

    # # subprocess.Popen(["python", "util.py"])
    worker_processes = []

    for _ in range(num_workers):
        p = Process(target=run_worker)
        p.start()
        worker_processes.append(p)
    # Start Flask application
    app.run(host='0.0.0.0', port=8868)

    # Join the worker processes
    for p in worker_processes:
        p.join()
