import openai
import os
import psycopg2
import uuid
import sys
from dotenv import load_dotenv
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'PaddleOCR2Pytorch')))
from PaddleOCR2Pytorch.pytorchocr.utils.utility import get_image_file_list_request
from PaddleOCR2Pytorch.tools.infer.pytorchocr_utility import  parse_args
from PaddleOCR2Pytorch.tools.infer.predict_system import TextSystem
import redis
import logging
load_dotenv()

def generate_uuid_v4():
    return str(uuid.uuid4())
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'postgres')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
redis_host = os.getenv('REDIS_HOST','redis' )
redis_port = os.getenv('REDIS_PORT', '6379')
OpenAIKey = os.getenv('OpenAIKey', None)
rec_image_shape = os.getenv('rec_image_shape', None)
det_yaml_path = os.getenv('det_yaml_path', None)
rec_yaml_path = os.getenv('rec_yaml_path', None)
rec_char_dict_path = os.getenv('rec_char_dict_path', None)
rec_model_path = os.getenv('rec_model_path', None)
det_model_path = os.getenv('det_model_path', None)
cls_model_path = os.getenv('cls_model_path', None)

redis_client = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def get_ada_embedding(text):
    openai.api_key = OpenAIKey
    response = openai.Embedding.create(
        input=text,
        engine="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    return embedding


def retrieve_nearest_embedding(query_embedding):
    conn = psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT
    )
    with conn.cursor() as cur:
        query_embedding_str = '[' + ', '.join(map(str, query_embedding)) + ']'
        cur.execute("""
            SELECT id, content, embedding <=> %s::vector AS distance
            FROM items
            ORDER BY distance
            LIMIT 1
        """, (query_embedding_str,))
        result = cur.fetchone()

    conn.commit()
    conn.close()
    return result


def process_ocr_and_store(url,job_id):
    logger.info(f"Processing OCR for URL: {url}, Job ID: {job_id}")
    redis_client.set(job_id, 'processing')
    try:
        args = parse_args()
        args.rec_image_shape = rec_image_shape
        args.det_yaml_path = det_yaml_path
        args.rec_yaml_path = rec_yaml_path
        args.rec_char_dict_path = rec_char_dict_path
        # path model converter
        args.rec_model_path = rec_model_path
        args.det_model_path = det_model_path
        args.cls_model_path = cls_model_path
        embedding_text = []
        image_file_list = get_image_file_list_request(url)
        text_sys = TextSystem(args)
        for img in image_file_list[0:2]:
            dt_boxes, rec_res = text_sys(img)
            result_string = " ".join(text for text, _ in rec_res)
            embedding = get_ada_embedding(result_string)
            embedding_text.append({"embedding": embedding, "text": result_string})

        conn = psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
            if not cur.fetchone():
                cur.execute("CREATE EXTENSION vector;")

            cur.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id UUID PRIMARY KEY,
                embedding VECTOR(1536),
                content TEXT
            );
            """)
            for v in embedding_text:
                _id = generate_uuid_v4()
                cur.execute("INSERT INTO items (id, embedding, content) VALUES (%s, %s, %s);", (_id, v["embedding"], v["text"]))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error during OCR processing: {e}")
    finally:
        conn.close()
        redis_client.set(job_id, 'done')