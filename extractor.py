# =========================
# extractor.py
# =========================
import tempfile
import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv


import fitz  
import base64
import os
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()


class PDFExtractor:
    def __init__(self):
        """Initialize the Docling PDF converter."""

    def extract(self, pdf_file) -> List[Dict]:
        """
        Main extraction method.
        Returns a list of dicts with 'page' and 'structured_content' keys.
        Compatible with existing chunking/embedding workflow.
        """
        extracted_pages = []
        pages_b64 = pdf_to_base64_pages(pdf_file)

        try:
            # Step 3: Initialize Hugging Face InferenceClient
            client = InferenceClient(
                provider="novita",
                api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],  
            )


            prompt="""You are an intelligent image analysis assistant who can extract information from the images also those images that are embedded within the image. Your task is to extract information from images as accurately and completely as possible. You should detect text, tables, and embedded images, and provide concise descriptions for regular images. Always follow the output format strictly, even if some sections are empty.



            Analyze the provided image and extract information according to the following rules:



            1.Textual Information:

            (In this section you will provide the textual information as it is in the picture)

            Extract all visible text exactly as it appears in the image.

            Preserve line breaks, punctuation, and order. if it is double columned provide in single column.



            2.Tables:


            (In this section you will provide all tables also those tables that are in embedded images)

            Extract all tables in a structured format.

            Include tables that are embedded inside images or diagrams.

            If tables have titles, include them.(If it doesn't have any title provide one according to table information)




            3.Textual Information in Embedded Images:

            (In this section you will provide all the textual information that are in embedded images with numbering serially.)

            Detect any text present within images inside the main image (e.g., charts, signs, screenshots).

            Extract the text exactly as it appears.





            4.Images Short Description:

            (In this section you will provide the short descriptions of the regular images.)

            For any regular embedded images, provide a very short description (e.g., “Image 1: a bar chart(short description),” “Image 2: a landscape photo(short description)”).




            Important Rules:

            Always output all four sections, even if some are blank.Just provide "None" there.

            Do not merge sections.

            Do not add extra explanations or commentary.

            Preserve the exact structure.

            """



            # Step 4: Send pages one by one to model
            for i, page_b64 in enumerate(pages_b64, start=1):
                completion = client.chat.completions.create(
                    model="Qwen/Qwen3-VL-235B-A22B-Instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": page_b64}
                                }
                            ]
                        }
                    ],
                )
                extracted_pages.append({
                    "page": i,
                    "structured_content": completion.choices[0].message.content
                })
                

            return extracted_pages

        except Exception as e:
            st.error(f"PDF extraction failed: {e}")
            return []

        finally:
            pages_b64 = []  # Clear to free memory
            


# Convert PDF pages to base64 images
def pdf_to_base64_pages(pdf_path, dpi=150, img_format="png"):
    try:
        doc = fitz.open(pdf_path)
        base64_pages = []

        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes(img_format)
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            base64_pages.append(f"data:image/{img_format};base64,{img_b64}")

        return base64_pages
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
        return []