# PDF-based RAG Pipeline and Clustering-based Non-text Element Extraction System
#
# This project extracts both text and non-text elements (images, shapes, etc.) from PDFs,
# integrating them into a Retrieval Augmented Generation (RAG) Q&A pipeline and image captioning system.
# It features custom clustering instead of traditional OCR, storing results in Milvus (vector DB) and MySQL.

# ------------------------------------------------------------------------------
# Key Features
# ------------------------------------------------------------------------------

# - Upload PDFs via Flask web interface
# - Extracts text and non-text (image, shape) elements
# - Custom clustering merges adjacent elements into grouped regions
# - Regions saved as PDF/PNG and recorded in MySQL
# - Caption (e.g., "Figure", "Table") detection + DFS-based 1:1 matching
# - LangChain + Milvus RAG Q&A with HuggingFace LLMs (e.g., Mixtral)
# - Salesforce BLIP model for automatic image captioning

# ------------------------------------------------------------------------------
# Example Paper
# ------------------------------------------------------------------------------
# File: ./data/4.pdf
# Hierarchical clustering analyzed using dual reward objectives (Dasgupta 2016)
# Complex visuals (lines, shapes) demonstrate limits of OCR-based AI models

# ------------------------------------------------------------------------------
# System Architecture
# ------------------------------------------------------------------------------

# Backend:
#   - Flask: Web server, chat, file upload
#   - MySQL: Stores PDFs, captions, regions
#   - Milvus: Vector DB for embeddings
#   - Docker: Hosts Milvus

# PDF Processing (c.py):
#   - PyMuPDF (fitz): Reads PDFs, extracts images/shapes
#   - Custom Clustering: Groups nearby visual elements
#   - Stores clustered regions as PNG/PDF + metadata in SQL

# RAG Pipeline:
#   - LangChain: Manages chunks and search
#   - HuggingFace (Mixtral): Answers Q&A

# Image Captioning:
#   - Salesforce BLIP: Converts visuals into English descriptions

# ------------------------------------------------------------------------------
# Installation & Execution
# ------------------------------------------------------------------------------

# Step 1: Install dependencies
pip install flask flask-cors mysql-connector-python python-dotenv pymupdf pillow transformers langchain

# Step 2: Make sure MySQL and Milvus are set up
# (You can use Docker for Milvus)

# Step 3: Set up .env file with credentials (API key, DB info)

# Step 4: Run Flask Web Server (file upload & Q&A)
flask run --debug

# Step 5: Process PDFs (non-text extraction and clustering)
python app.py
# OR
python c.py

# Step 6: Run image captioning test
python image_caption.py

# ------------------------------------------------------------------------------
# SQL Schema
# ------------------------------------------------------------------------------

# Table: area
# Stores clustered regions (figures/tables) and visual metadata
# Linked to captions table via caption_id

# Table: captions
# Stores text captions with location and PDF mapping

# Table: pdf_documents
# Tracks each uploaded PDF and processing timestamp

# Run in MySQL:
mysql -u root -p
# Paste the SQL schema below:

# CREATE TABLE `area` (
#   `area_id` int NOT NULL AUTO_INCREMENT,
#   `caption_id` int DEFAULT NULL,
#   `pdf_file_name` text,
#   `png_file_name` text,
#   `page_number` int NOT NULL,
#   `x0` double DEFAULT NULL,
#   `y0` double DEFAULT NULL,
#   `x1` double DEFAULT NULL,
#   `y1` double DEFAULT NULL,
#   `type` enum('figure','table') NOT NULL,
#   `appearance_description` text,
#   PRIMARY KEY (`area_id`),
#   KEY `caption_id` (`caption_id`),
#   CONSTRAINT `area_ibfk_1` FOREIGN KEY (`caption_id`) REFERENCES `captions` (`caption_id`) ON DELETE SET NULL
# ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

# CREATE TABLE `captions` (
#   `caption_id` int NOT NULL AUTO_INCREMENT,
#   `caption_name` text,
#   `pdf_id` int NOT NULL,
#   `page_number` int NOT NULL,
#   `caption_text` text,
#   `x0` double DEFAULT NULL,
#   `y0` double DEFAULT NULL,
#   `x1` double DEFAULT NULL,
#   `y1` double DEFAULT NULL,
#   PRIMARY KEY (`caption_id`),
#   KEY `pdf_id` (`pdf_id`),
#   CONSTRAINT `captions_ibfk_1` FOREIGN KEY (`pdf_id`) REFERENCES `pdf_documents` (`pdf_id`) ON DELETE CASCADE
# ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

# CREATE TABLE `pdf_documents` (
#   `pdf_id` int NOT NULL AUTO_INCREMENT,
#   `file_name` text NOT NULL,
#   `processed_date` datetime DEFAULT CURRENT_TIMESTAMP,
#   PRIMARY KEY (`pdf_id`)
# ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

# ------------------------------------------------------------------------------
# Usage Guide
# ------------------------------------------------------------------------------

# - Open browser and go to Flask app URL
# - Upload PDF via interface
# - System processes and clusters visual elements
# - Saved files: output/pdf, output/png
# - Data stored in SQL and Milvus
# - Ask questions via chat, get AI-generated answers from PDF context

# ------------------------------------------------------------------------------
# Future Improvements
# ------------------------------------------------------------------------------

# - Better Q&A accuracy via stronger models and prompt engineering
# - UI/UX enhancements
# - Support for DOCX, PPTX, and more formats
# - Faster and more accurate clustering
# - Dockerized deployment to cloud

# ------------------------------------------------------------------------------
# Conclusion
# ------------------------------------------------------------------------------

# This project presents a novel non-OCR method to analyze visual info in PDFs.
# It allows deep understanding of complex academic documents by combining:
#  - Visual element clustering
#  - Caption-text matching
#  - Natural language Q&A
#  - AI-generated image descriptions

# Ideal for research, portfolio, or demo projects in AI document understanding.

# ------------------------------------------------------------------------------

# 👉 Visit https://gptonline.ai/ko/ to explore more AI document tools!
