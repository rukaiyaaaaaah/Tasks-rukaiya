import torch
import pandas as pd
import base64
from io import StringIO
import logging
import gradio as gr
from transformers import pipeline
import warnings
from importlib import resources
import sys
import traceback

# Setup logging
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')

# Initialize the tokenizer and model globally to avoid reloading them every time the function is called
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline("token-classification", model="Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2", device=device)
columns_to_redact = []

def process_column(texts):
    processed_texts = []
    for text in texts:
        model_output = pipe(text)
        merged_entities = []
        for i, token in enumerate(model_output):
            if i == 0 or (model_output[i-1]['end'] == token['start'] and model_output[i-1]['entity'] == token['entity']):
                if merged_entities and model_output[i-1]['entity'] == token['entity']:
                    merged_entities[-1]['word'] += text[token['start']:token['end']]
                    merged_entities[-1]['end'] = token['end']
                else:
                    merged_entities.append(token.copy())
                    merged_entities[-1]['word'] = text[token['start']:token['end']]
            else:
                merged_entities.append(token.copy())
                merged_entities[-1]['word'] = text[token['start']:token['end']]
        for entity in merged_entities:
            text = text.replace(entity['word'], f"[REDACTED {entity['entity']}]")
        processed_texts.append(text)
    return processed_texts

def modify_csv(file, columns_to_redact):
    try:
        df = pd.read_csv(file)
        for column_name in columns_to_redact:
            if column_name in df.columns:
                df[column_name] = process_column(df[column_name].astype(str))
        output = StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return df
    except Exception as exc:
        logging.error(f"Error processing columns: {exc}", exc_info=True)

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def hide_all_warnings():
    """Suppress all warnings."""
    warnings.filterwarnings("ignore")

hide_all_warnings()

def display_csv(file):
    df = pd.read_csv(file)
    return df

def export_csv(d):
    d.to_csv("output.csv")
    return gr.File(value="output.csv", visible=True)

custom_css = """
body { background-color: #1f1f1f; }
div { color: white; }
h1 { text-align: center; color: #ffffff; }
label { color: #ffffff; }
input, textarea { background-color: #333333; border-color: #555555; color: white; }
"""

def gradio_interface(file):
    global columns_to_redact
    return modify_csv(file, columns_to_redact)

def launch_gradio():
    logo_path = "/Users/rukaiyahhasan/Kavach/Kavach/data/logo.png"
    logo_base64 = image_to_base64(logo_path)

    with gr.Blocks(css=custom_css) as demo:
        with gr.Row():
            gr.Markdown(f"""
                <div style='background-color: #f0eae3; padding: 10px; text-align: center; border-radius: 10px; margin-bottom: 90px;'>
                    <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="height: 100px; width: 100px; display: block; margin-left: auto; margin-right: auto;">
                </div>
            """)
        with gr.Column():
            csv_input = gr.File(label="Upload CSV File", file_types=['.csv'])
            csv_output = gr.Dataframe(type="pandas", col_count=7)
            button = gr.Button("Export")
            csv = gr.File(interactive=False, visible=False)

        button.click(export_csv, csv_output, csv)
        csv_input.change(gradio_interface, inputs=csv_input, outputs=csv_output)

        demo.launch()

def main():
    global columns_to_redact

    try:
        input_columns = input("Enter column names to redact, separated by commas: ")
        columns_to_redact.extend([x.strip() for x in input_columns.split(',')])
        launch_gradio()
    except Exception as e:
        logging.error(f"Failed to launch the interface: {e}", exc_info=True)
        traceback.print_exc()  # This line prints detailed traceback information to the console

        print("An error occurred, please try again later.")

if __name__ == "__main__":
    main()
