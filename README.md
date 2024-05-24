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
base64_image=$(base64 -i  /Users/huongtruong/Documents/huongtruong_project/Repo_Git/PaddleOCR2Pytorch/test.jpeg)
curl -H "Content-Type:application/json" -X POST --data "{\"base64_data\": [\"$base64_image\"]}" http://localhost:8868/OCR
```
#### 
```bash
base64_pdf=$(base64 -i  /Users/huongtruong/Documents/huongtruong_project/Repo_Git/PaddleOCR2Pytorch/t.pdf)
echo "{\"base64_data\": [\"$base64_pdf\"]}" > data.json
curl -H "Content-Type: application/json" -X POST --data @data.json http://localhost:8868/OCR
```
### Request chatbot
```bash
curl -X POST http://localhost:8868/chat -H "Content-Type: application/json" -d '{"input": "What is ID Docusign Envelope"}'
```# RAG
