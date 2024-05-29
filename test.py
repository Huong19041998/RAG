import requests
from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import io
from PIL import Image
import numpy as np
import cv2
import mimetypes

def is_image(url):
    mimetypes.init()
    response = requests.head(url)
    content_type = response.headers.get('content-type')
    if content_type and 'image' in content_type:
        return True
    return False

def is_pdf(url):
    mimetypes.init()
    response = requests.head(url)
    content_type = response.headers.get('content-type')
    if content_type and 'application/pdf' in content_type:
        return True
    return False


def download_image(image_url):
    response = requests.get(image_url)
    image_data = response.content
    image = Image.open(io.BytesIO(image_data))
    img_path = "img/test_.jpg"
    image.save(img_path)
    return img_path
def download_pdf(url):
    response = requests.get(url)
    data = response.content
    # Mở tài liệu PDF từ dữ liệu đã tải về
    pdf_document = fitz.open("pdf", data)
    images = []
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()

        # Chuyển đổi pixmap thành numpy array
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Nếu hình ảnh có alpha channel (RGBA), chuyển đổi sang RGB
        if pix.n == 4:
            img_array = img_array[:, :, :3]
        images.append(img_array)
        cv2.imwrite("img/test.jpg",img_array)
url = "https://arxiv.org/pdf/2403.11703v1"
