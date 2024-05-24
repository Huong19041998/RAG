import openai

def get_ada_embedding(text,OpenAIKey):
    openai.api_key = OpenAIKey
    response = openai.Embedding.create(
        input=text,
        engine="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    return embedding


def retrieve_nearest_embedding(query_embedding,conn):
    # conn = psycopg2.connect(
    #     dbname=POSTGRES_DB,
    #     user=POSTGRES_USER,
    #     password=POSTGRES_PASSWORD,
    #     host=POSTGRES_HOST,
    #     port=POSTGRES_PORT
    # )
    with conn.cursor() as cur:
        # Chuyển vector truy vấn thành định dạng chuỗi
        query_embedding_str = '[' + ', '.join(map(str, query_embedding)) + ']'
        cur.execute("""
            SELECT id, content, embedding <=> %s::vector AS distance
            FROM items
            ORDER BY distance
            LIMIT 1
        """, (query_embedding_str,))
        result = cur.fetchone()
        print("result:____", result)
    conn.commit()
    conn.close()
    return result