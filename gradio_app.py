# import gradio as gr
# import os
# from classification_function import classify_document_type
# from extraction_function import extract_invoice_data, extract_bank_statement_data, extract_receipt_data
# from rag_pipeline_function import extract_text, clean_text, split_text, create_vectorstore, setup_qa_chain

# qa_chain = None

# def process_file(file):
#     file_path = file.name
#     doc_type = classify_document_type(file_path)
#     result = f"**Predicted Document Type**: `{doc_type}`\n\n"
#     df = None
#     if doc_type == "Invoice":
#         df = extract_invoice_data(file_path)
#     elif doc_type == "Bank Statement":
#         df = extract_bank_statement_data(file_path)
#     elif doc_type == "Money Receipt":
#         df = extract_receipt_data(file_path)

#     if df is not None and not df.empty:
#         result += f"**Extracted Data:**\n\n{df.to_markdown(index=False)}"
#     else:
#         result += "No relevant data extracted."
#     return result, gr.update(visible=True)

# def prepare_qa_pipeline(file):
#     global qa_chain
#     file_path = file.name
#     text = extract_text(file_path)
#     cleaned = clean_text(text)
#     chunks = split_text(cleaned)

#     vectorstore = create_vectorstore(chunks)
#     qa_chain = setup_qa_chain(vectorstore)
#     return "QA pipeline ready.", gr.update(visible=True)

# def answer_question(question):
#     if qa_chain is None:
#         return "QA pipeline not ready."
#     return f"**Answer:** {qa_chain.run(question).strip()}"

# with gr.Blocks(title="AI-Based Document Classification and Extraction System") as demo:
#     gr.Markdown("# 📄 AI-Based Document Classification and Extraction System")
#     gr.Markdown("Upload a document to classify, extract data, and ask questions.")

#     with gr.Row():
#         with gr.Column():
#             file_input = gr.File(label="Upload PDF or Image", type="filepath", file_types=[".pdf", ".jpg", ".jpeg", ".png"], file_count="single")
#             process_btn = gr.Button("Classify & Extract")
#             rag_btn = gr.Button("Build QA Pipeline")
#         with gr.Column():
#             extraction_output = gr.Markdown("Awaiting file...", visible=True)
#             qa_status_output = gr.Markdown("", visible=False)

#     with gr.Row(visible=False) as qa_section:
#         with gr.Column():
#             question_input = gr.Textbox(label=" Ask a Question", placeholder="e.g. What is the total amount?")
#             ask_btn = gr.Button(" Get Answer")
#         with gr.Column():
#             answer_output = gr.Markdown("")

#     process_btn.click(fn=process_file, inputs=file_input, outputs=[extraction_output, qa_status_output])
#     rag_btn.click(fn=prepare_qa_pipeline, inputs=file_input, outputs=[qa_status_output, qa_section])
#     ask_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)

# demo.launch(debug=True)

import gradio as gr
import os
from classification_function import classify_document_type
from extraction_function import extract_invoice_data, extract_bank_statement_data, extract_receipt_data
from rag_pipeline_function import extract_text, clean_text, split_text, create_vectorstore, setup_qa_chain

qa_chain = None

def process_file(file):
    file_path = file.name
    doc_type = classify_document_type(file_path)
    result = f"**Predicted Document Type**: `{doc_type}`\n\n"
    df = None
    if doc_type == "Invoice":
        df = extract_invoice_data(file_path)
    elif doc_type == "Bank Statement":
        df = extract_bank_statement_data(file_path)
    elif doc_type == "Money Receipt":
        df = extract_receipt_data(file_path)

    if df is not None and not df.empty:
        result += f"**Extracted Data:**\n\n{df.to_markdown(index=False)}"
    else:
        result += "No relevant data extracted."
    return result, gr.update(visible=True)

def prepare_qa_pipeline(file):
    global qa_chain
    file_path = file.name
    text = extract_text(file_path)
    cleaned = clean_text(text)
    chunks = split_text(cleaned)

    vectorstore = create_vectorstore(chunks)
    qa_chain = setup_qa_chain(vectorstore)
    return "✅ QA pipeline ready.", gr.update(visible=True)

def answer_question(question):
    if qa_chain is None:
        return "QA pipeline not ready."
    return f"**Answer:** {qa_chain.run(question).strip()}"

with gr.Blocks(css=".scroll-box { max-height: 300px; overflow-y: auto; }", title="AI-Based Document Classification and Extraction System") as demo:
    gr.Markdown("# 📄 AI-Based Document Classification and Extraction System")
    gr.Markdown("Upload a document to classify, extract data, and ask questions.")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload PDF or Image", type="filepath", file_types=[".pdf", ".jpg", ".jpeg", ".png"], file_count="single")
            process_btn = gr.Button("Classify & Extract")
            rag_btn = gr.Button("Build QA Pipeline")
        with gr.Column():
            extraction_output = gr.Markdown("Awaiting file...", visible=True, elem_classes=["scroll-box"])
            qa_status_output = gr.Markdown("", visible=False)

    with gr.Row(visible=False) as qa_section:
        with gr.Column():
            question_input = gr.Textbox(label=" Ask a Question", placeholder="e.g. What is the total amount?")
            ask_btn = gr.Button(" Get Answer")
        with gr.Column():
            answer_output = gr.Markdown("")

    process_btn.click(fn=process_file, inputs=file_input, outputs=[extraction_output, qa_status_output])
    rag_btn.click(fn=prepare_qa_pipeline, inputs=file_input, outputs=[qa_status_output, qa_section])
    ask_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)

demo.launch(server_name="0.0.0.0",debug=True)
