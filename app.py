from flask import Flask, render_template, request, redirect, url_for, session, jsonify,send_from_directory, abort
from flask_cors import CORS
import os
from logic.c import process_pdf

from logic.lang_pipe_line import build_rag_pipeline_for_pdf_id, execute_question, serialize_document
from logic.lang_delete import initialize_model
import json
app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False

# 업로드 폴더, 허용 확장자, 출력 폴더 설정
app.config['UPLOAD_FOLDER'] = 'data'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['OUTPUT_DIR'] = 'clustered'
app.secret_key = 'some_secret_key'  # session 사용을 위한 secret key

# 전역 변수: 업로드된 PDF에 해당하는 pdf_id와 RAG 체인
global_vactor_store = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    initialize_model()
    global global_vactor_store  # 전역 변수를 업데이트
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
            global_vactor_store = build_rag_pipeline_for_pdf_id(pdf_id)
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
        if global_vactor_store is None:
            return jsonify({"error": "RAG 체인이 아직 구축되지 않았습니다."})
        
        response = execute_question(global_vactor_store, query)
        if "context" in response:
            response["context"] = [serialize_document(doc) for doc in response["context"]]
        # print("응답 결과:")
        
        # print(json.dumps(response, indent=2, ensure_ascii=False))

        answer = response.get("answer", "답변이 생성되지 않았습니다.")
        print(response)

        pdf_results = global_vactor_store.similarity_search(query, k=1)
        for doc in pdf_results:
            print(doc.metadata)
            # print(f"내용2: {doc.page_content}")
            # print(f"pdf_file_name: {doc.metadata.get('pdf_file_name', '')}")
            # print(f"caption_name: {doc.metadata.get('caption_name', '')}")
            print(f"appearance_description: {doc.metadata.get('source', '')}")
            print("---")


        return jsonify({"input": query, "answer": answer,"pdf_file_name": pdf_results[0].metadata.get('source', '')})
    return render_template('chat.html')



@app.route('/download')
def download_file():
    file_path = request.args.get('file')
    if not file_path:
        return abort(400, "파일 경로가 지정되지 않았습니다.")
    
    # 파일 경로가 안전한지 검증하는 로직 필요 (예: os.path.join 사용)
    directory = os.path.abspath(os.path.join(os.getcwd(), os.path.dirname(file_path)))
    filename = os.path.basename(file_path)

    # 파일 존재 여부 확인
    if not os.path.exists(os.path.join(directory, filename)):
        return abort(404, "파일을 찾을 수 없습니다.")
    
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    app.run(host='192.168.0.106', port=5000, debug=True)
