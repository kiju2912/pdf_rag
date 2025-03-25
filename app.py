from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
import os
from logic.c import process_pdf
from logic.lang_pipe_line import build_rag_pipeline_for_pdf_id, execute_question
from logic.lang_delete import initialize_model

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False

# 업로드 폴더, 허용 확장자, 출력 폴더 설정
app.config['UPLOAD_FOLDER'] = 'data'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['OUTPUT_DIR'] = 'clustered'
app.secret_key = 'some_secret_key'  # session 사용을 위한 secret key

# 전역 변수: 업로드된 PDF에 해당하는 pdf_id와 RAG 체인
global_rag_chain = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    initialize_model()
    global global_rag_chain  # 전역 변수를 업데이트
    if request.method == 'POST':
        if 'pdf' not in request.files:
            return render_template('index.html', error="파일이 첨부되지 않았습니다.")
        file = request.files['pdf']
        if file.filename == '':
            return render_template('index.html', error="파일명이 없습니다.")
        if file and allowed_file(file.filename):
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            if not os.path.exists(app.config['OUTPUT_DIR']):
                os.makedirs(app.config['OUTPUT_DIR'])
            output_filepath = os.path.join(app.config['OUTPUT_DIR'], file.filename)
            if os.path.exists(output_filepath):
                os.remove(output_filepath)

            # process_pdf 함수가 pdf_id를 반환한다고 가정
            pdf_id = process_pdf(filepath, output_filepath)
            session['pdf_id'] = pdf_id
            session['file_name'] = file.filename

            # 전역 변수에 RAG 체인 업데이트
            global_rag_chain = build_rag_pipeline_for_pdf_id(pdf_id)
            return redirect(url_for('chat'))
        else:
            return render_template('index.html', error="PDF 파일만 업로드할 수 있습니다.")
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        query = request.form.get("query")
        if not query:
            return jsonify({"error": "쿼리를 입력해주세요."})
        if 'pdf_id' not in session:
            return jsonify({"error": "PDF가 업로드되지 않았습니다."})
        if global_rag_chain is None:
            return jsonify({"error": "RAG 체인이 아직 구축되지 않았습니다."})
        
        response = execute_question(global_rag_chain, query)
        answer = response.get("answer", "답변이 생성되지 않았습니다.")
        return jsonify({"input": query, "answer": answer})
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(host='192.168.0.106', port=5000, debug=True)
