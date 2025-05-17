from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from uuid import uuid4
import fitz  # PyMuPDF
import requests
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import torch
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

# Load OpenAI API key from environment variable

app = FastAPI()
templates = Jinja2Templates(directory="templates")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# session_id => {"chunks": [...], "embeddings": tensor, "chat": [(q,a),...]}
session_store = {}

def extract_text_from_pdf_stream(stream: BytesIO) -> str:
    doc = fitz.open(stream=stream, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def embed_text(chunks):
    return embed_model.encode(chunks, convert_to_tensor=True)

def search_chunks(embeddings, chunks, question):
    question_embedding = embed_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, embeddings)[0]
    top_results = torch.topk(similarities, k=min(3, len(chunks)))
    return [chunks[i] for i in top_results.indices]

def generate_answer(context: str, question: str) -> str:
    try:
        prompt = f"""You are a helpful assistant. Use the following document context to answer the question.

Document Context:
\"\"\"
{context}
\"\"\"

Question: {question}
Answer:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating answer: {e}"

@app.get("/", response_class=HTMLResponse)
async def get_upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/load", response_class=HTMLResponse)
async def load_pdf(
    request: Request,
    pdf_url: str = Form(""),
    file: UploadFile = File(None)
):
    session_id = str(uuid4())
    try:
        if pdf_url:
            response = requests.get(pdf_url)
            if response.status_code != 200:
                raise Exception("Failed to download PDF from URL.")
            text = extract_text_from_pdf_stream(BytesIO(response.content))
        elif file:
            content = await file.read()
            text = extract_text_from_pdf_stream(BytesIO(content))
        else:
            raise Exception("No PDF provided.")

        chunks = [chunk for chunk in text.split("\n\n") if len(chunk.strip()) > 30]
        embeddings = embed_text(chunks)

        session_store[session_id] = {
            "chunks": chunks,
            "embeddings": embeddings,
            "chat": []
        }

        return templates.TemplateResponse("chat.html", {
            "request": request,
            "session_id": session_id,
            "chat_history": [],
        })

    except Exception as e:
        return templates.TemplateResponse("upload.html", {"request": request, "error": str(e)})

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, session_id: str = Form(...), message: str = Form(...)):
    message = message.strip()
    chat_data = session_store.get(session_id)

    if not chat_data:
        return HTMLResponse("Invalid session.")

    if message.lower() in ["stop", "thank you", "thanks", "bye"]:
        return templates.TemplateResponse("upload.html", {"request": request, "message": "Chat ended. Upload new document to start again."})

    relevant_chunks = search_chunks(chat_data["embeddings"], chat_data["chunks"], message)
    context = "\n".join(relevant_chunks)
    answer = generate_answer(context, message)

    chat_data["chat"].append((message, answer))

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "session_id": session_id,
        "chat_history": chat_data["chat"]
    })
