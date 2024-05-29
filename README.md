# Structure
```
project_folder/
│
├── README.md
├── app.py
├── util.py
├── model/
│   ├── ch_ptocr_mobile_v2.0_cls_infer.pth
│   ├── en_ptocr_v3_det_infer.pth
│   ├── en_ptocr_v4_rec_infer.pth
│   
├── PaddleOCR2Pytorch
├── pgvector
```
# Run 
### Copy model ocr  to model
### Open docker-compose.yml paste OpenAIKey

### Build Docker 

```bash
docker-compose build
docker-compose up -d

```
### Request OCR and Insert Data to DB
#### Images 
```bash
curl -H "Content-Type:application/json" -X POST --data "{\"url\": [\"url_img\"]}" http://localhost:8868/OCR

```
#### 
```bash

curl -H "Content-Type:application/json" -X POST --data "{\"url\": [\"https://arxiv.org/pdf/2403.11703v1\"]}" http://localhost:8868/OCR

```

### Request chatbot
```bash
curl -X POST http://localhost:8868/chat -H "Content-Type: application/json" -d '{"input": "What is ID Docusign Envelope"}'
```
