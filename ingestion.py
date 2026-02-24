import pdfplumber
import os

def load_documents(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)

            with pdfplumber.open(full_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""

            documents.append({
                "text": text,
                "source": file
            })

    return documents