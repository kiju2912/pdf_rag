# PDF-based RAG Pipeline and Clustering-Based Non-Text Element Extraction System

This project is a platform that effectively extracts and analyzes text and non-text (images, shapes, etc.) information from PDF documents and integrates it with a Retrieval Augmented Generation (RAG) Q&A system and image captioning functionality. A key feature is the extraction of non-text elements as data using a custom clustering technique instead of traditional OCR, storing this data in both a vector store and SQL database for further utilization.

---

## Key Features

- **PDF Upload and Processing**  
  - Users can upload PDF files via a **Flask web interface**, which are then saved to a designated folder for text and non-text extraction.
  - Using a **custom `c.py` module**, non-text elements (images, drawings, etc.) in the PDF are extracted based on coordinates and clustered/grouped into single regions through a proximity-based clustering algorithm.
  - Each cluster region is saved as a separate PDF and PNG file, with the corresponding metadata stored in an SQL database for future analysis and retrieval.

- **Text and Caption Analysis**  
  - Text blocks in the PDF are analyzed to detect captions (e.g., Figure, Table), which are matched 1:1 with clustered regions using a DFS-based matching algorithm.
  - Extracted captions and regions are visually annotated in the PDF, with a final PDF output containing clustering and caption data.

- **RAG Pipeline and Q&A System**  
  - Utilizes **LangChain** and **Milvus vector store** to embed text from the PDF and additional data (captions, non-text regions), allowing the system to retrieve relevant context in response to natural language questions.
  - **HuggingFace language models** (e.g., `mistralai/Mixtral-8x7B-Instruct-v0.1`) are used to generate answers, referencing matched figure and cluster information.

- **Image Captioning**  
  - Applies **Salesforce’s BLIP model** to generate automatic captions for PNG images extracted from non-text regions.
  - These captions supplement the visual content of the extracted areas and are stored alongside metadata in the SQL database.

---

## Example Paper

![4.pdf](./data/4.pdf)

- **Main Content**
   - While hierarchical clustering has long been used in data analysis, it lacks a clear objective function and performance guarantees. This paper introduces a dual reward objective based on Dasgupta (2016)’s cost objective, analyzing what goals algorithms optimize and comparing different methods.

- **Why It Was Chosen**
   - The paper explains concepts using non-textual elements (lines, circles, etc.), which traditional AI methods struggle to interpret or extract.

---

## System Architecture

- **Backend**  
  - **Flask**: Web server, file upload, session management, and Q&A chat interface.
  - **MySQL**: Stores PDF, caption, and region metadata.
  - **Milvus**: Vector store for embeddings and similarity search.
  - **Docker**: Used to host Milvus in a virtual Linux environment.

- **PDF Processing and Non-Text Extraction (c.py)**  
  - **PyMuPDF (fitz)**: Reads and manipulates PDF files, extracts image/shape regions.
  - **https://github.com/kiju2912/pdf_parser**
  - ![Processed PDF](./clustered/4.pdf)  
    - Analyzes text blocks and detects captions (figure/table).
    - Clusters and groups non-text elements (merging adjacent elements, overlapping regions, etc.)
    - Matches clusters with captions and stores them as separate PDF/PNG files and SQL records.

- **NLP and RAG Pipeline (lang_pipe_line.py)**  
  - **LangChain**: Constructs Q&A chains and document chunking.
  - **HuggingFace models**: Generates answers using models like `Mixtral-8x7B-Instruct-v0.1`.

- **Image Captioning (image_caption.py)**  
  - **BLIP Model (Salesforce)**: Converts extracted regions into images and describes them.
  - Example:  
    - File: Figure 2_1743059622915256000.png, Caption: *a diagram of a single-line network*

---

## Screenshots

These screenshots illustrate key features and the UI:

1. **Start Page**  
   ![Start Page](./readme/시작화면.png)  
   - Shows the main screen before PDF upload. Users can upload a file using the "Register PDF before starting chat" button.

2. **Chat Interface (Initial State)**  
   ![Chat Initial State](./readme/챗팅_기본화면.png)  
   - The initial state of the chatbot interface. Users type questions and receive answers in the chat window.

3. **Figure Question Example 1**  
   ![Figure Q1](./readme/figure에_관한-질문1.png)  
   - Asking about “Figure 4.” The system retrieves relevant information and provides an answer.
   - If the `source` value in metadata includes "output", the section is prepared for download.

4. **Located Figure 4**  
   ![Located Figure 4](./readme/탐색된figure1(figure4).png)  
   - Shows how “Figure 4” is found, extracted, and converted into a separate PDF.

5. **General Question**  
   ![General Q](./readme/일반질문.png)  
   - Example of a general question about the PDF content.

6. **Figure Question Example 2**  
   ![Figure Q2](./readme/figure에_관한_질문2.png)  
   - Asking about “Figure 13.” Download preparation is the same if metadata `source` includes "output".

7. **Located Figure 13**  
   ![Located Figure 13](./readme/탐색된figure2(figure13).png)  
   - Result of locating and extracting “Figure 13” as a separate PDF.

8. **Extracted Regions**  
   ![Extracted Regions](./readme/추출된_영역들.png)  
   - Visual output of extracted figures, tables, and captions from the PDF.

---

## Installation and Running

1. **Environment Setup & Dependencies**  
   - Create a virtual environment in Python 3.x and install required packages:
     ```bash
     pip install flask flask-cors mysql-connector-python python-dotenv pymupdf pillow transformers langchain
     ```
   - Ensure **Milvus** and **MySQL** are installed and configured beforehand.
   - Set HuggingFace API tokens and other environment variables in the `.env` file.

2. **Running the Project**  
   - **Flask Web Server (PDF Upload + Q&A Chat):**
     ```bash
     flask run --debug
     ```
   - **Run PDF Processing (c.py Module):**  
     ```bash
     python app.py  # or python c.py
     ```
     This processes PDF files in the `data` folder and stores output in `output/pdf` and `output/png`.

   - **Test Image Captioning:**
     ```bash
     python image_caption.py
     ```

3. **How to Use**  
   - Access the Flask app via a web browser and upload a PDF.
   - The system will extract and cluster text/non-text content, storing the results in PDFs/PNGs and SQL.
   - Captions and region metadata are saved for vector search and used in the RAG pipeline.
   - Ask questions in the chat interface, and the system will search contextually relevant information to generate an answer.

---

## SQL DB Schema

```sql
CREATE TABLE `area` (
  `area_id` int NOT NULL AUTO_INCREMENT,
  `caption_id` int DEFAULT NULL,
  `pdf_file_name` text,
  `png_file_name` text,
  `page_number` int NOT NULL,
  `x0` double DEFAULT NULL,
  `y0` double DEFAULT NULL,
  `x1` double DEFAULT NULL,
  `y1` double DEFAULT NULL,
  `type` enum('figure','table') NOT NULL,
  `appearance_description` text,
  PRIMARY KEY (`area_id`),
  KEY `caption_id` (`caption_id`),
  CONSTRAINT `area_ibfk_1` FOREIGN KEY (`caption_id`) REFERENCES `captions` (`caption_id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `captions` (
  `caption_id` int NOT NULL AUTO_INCREMENT,
  `caption_name` text,
  `pdf_id` int NOT NULL,
  `page_number` int NOT NULL,
  `caption_text` text,
  `x0` double DEFAULT NULL,
  `y0` double DEFAULT NULL,
  `x1` double DEFAULT NULL,
  `y1` double DEFAULT NULL,
  PRIMARY KEY (`caption_id`),
  KEY `pdf_id` (`pdf_id`),
  CONSTRAINT `captions_ibfk_1` FOREIGN KEY (`pdf_id`) REFERENCES `pdf_documents` (`pdf_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `pdf_documents` (
  `pdf_id` int NOT NULL AUTO_INCREMENT,
  `file_name` text NOT NULL,
  `processed_date` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`pdf_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

## Future Improvements

- **Improve QA Model Accuracy**  
  - Enhance the quality and relevance of generated answers by incorporating more advanced large language models (LLMs) and applying refined prompt engineering techniques.

- **UI/UX Enhancement**  
  - Develop a more intuitive and modern frontend interface.
  - Implement real-time feedback and visualizations for processed PDF data.

- **Support for More File Formats**  
  - Extend support to other document formats beyond PDF (e.g., Word, PowerPoint, HTML).
  - Enable integration with cloud-based storage or document platforms.

- **Clustering Algorithm Optimization**  
  - Refine the clustering logic for non-text elements to increase speed and precision.
  - Improve handling of overlapping and nested visual elements.

- **Deployment and Scalability**  
  - Containerize the system using Docker and Kubernetes for scalable deployment.
  - Prepare for cloud deployment (e.g., AWS, GCP) to ensure production readiness.

---

## Conclusion

This project proposes a new method for analyzing PDF documents by structurally extracting and clustering non-text elements without relying on traditional OCR techniques. By integrating textual and visual information into a unified system, it enables in-depth analysis of academic papers and complex documents. Coupled with a natural language Q&A pipeline and image captioning system, the platform offers a rich and interactive way to explore PDF contents. This makes it suitable for use in research demos, academic projects, and intelligent document processing tools.


