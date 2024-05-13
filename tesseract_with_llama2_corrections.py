from pdf2image import convert_from_path
import pytesseract
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import hashlib
import sqlite3
import numpy as np
from os import getenv
import time
import psutil


def convert_pdf_to_images_func(input_pdf_file_path, max_test_pages):
    if max_test_pages == 0:
        print(f"Now converting all pages of PDF file {input_pdf_file_path} to images...")
        list_of_scanned_images = convert_from_path(input_pdf_file_path, first_page=1)
    else:
        print(f"Now converting first {max_test_pages} pages of PDF file {input_pdf_file_path} to images...")
        list_of_scanned_images = convert_from_path(input_pdf_file_path, first_page=1, last_page=max_test_pages)
    print(f"Done converting pages from PDF file {input_pdf_file_path} to images.")
    return list_of_scanned_images

def ocr_image(image):
    return pytesseract.image_to_string(image)
    
def check_extracted_pages_func(extracted_text_string):
    #first check if it's long enough to be a page:
    if len(extracted_text_string) < 10:
        return False
    #now check if it has enough words to be a page:
    if len(extracted_text_string.split()) < 5:
        return False
    return extracted_text_string

def remove_intro(llm_output_2_text):
    try:
        # Strip leading and trailing whitespace before splitting the lines
        lines = llm_output_2_text.strip().splitlines()
        # Skip the first line and the following blank line
        lines = lines[2:] if lines[1].strip() == '' else lines[1:]
        return '\n'.join(lines)
    except Exception as e:
        print(f"Exception in remove_intro: {e}")
        return llm_output_2_text

def process_text_with_llm(extracted_text_string, check_if_valid_english=False, reformat_as_markdown=True):
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=getenv("OPENROUTER_API_KEY"))
    model = "meta-llama/llama-3-70b-instruct"
    messages = [{"role": "system", "content": "You are an intelligent text processing assistant."}]

    llm_tokens = 0
    
    # Check if the text is valid English
    if check_if_valid_english:
        messages.append({"role": "user", "content": f"Is this valid English text? (y/n): ```{extracted_text_string}```"})
        response = client.chat.completions.create(model=model, messages=messages)
        llm_tokens += response.usage.total_tokens
        valid_english = response.choices[0].message.content.strip().lower() == 'y'
    else:
        valid_english = False

    # Correct any typos
    corrected_text = extracted_text_string
    if valid_english or not check_if_valid_english:
        messages.append({"role": "user", "content": f"Correct any typos in this text, using common sense reasoning and only respond with the corrected text : ```{extracted_text_string}```"})
        response = client.chat.completions.create(model=model, messages=messages)
        llm_tokens += response.usage.total_tokens
        corrected_text = response.choices[0].message.content

    # Reformat using markdown
    if reformat_as_markdown:
        messages.append({"role": "user", "content": f"Reformat this text to be more readable using markdown formatting and only respond with the markdown content: ```{corrected_text}```"})
        response = client.chat.completions.create(model=model, messages=messages)
        llm_tokens += response.usage.total_tokens
        corrected_text = response.choices[0].message.content

    return corrected_text, llm_tokens

def calculate_sentence_embedding_2(text):
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=getenv("OPENROUTER_API_KEY"))
    model = "text-embedding-ada-002"
    sentence_embedding = None
    while sentence_embedding is None:
        try:
            # Use the text-embedding-ada-002 model to compute embeddings
            response = client.Embeddings.create(
                model=model,
                input=text
            )
            sentence_embedding = response['data'][0]['embedding']
        except Exception as e:
            print(f"Exception in calculate_sentence_embedding: {e}")
            # Trimming the sentence if it's too long
            if "too many tokens" in str(e):
                text = text[:int(len(text) * 0.95)]
                print(f"Trimming sentence due to too many tokens. New length: {len(text)}")
            else:
                break
    return sentence_embedding

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def filter_hallucinations(corrected_text, raw_text, threshold=0.1, pdf_file_path=None, db_path=None):
    client = OpenAI()
    threshold_increment = 0.02
    original_embeddings, corrected_embeddings = None, None
    total_tokens = 0

    if db_path is not None and pdf_file_path is not None:
        # Check if the database file exists and compute hash of the pdf file
        if not os.path.isfile(db_path):
            print(f"No existing database found at {db_path}. Creating a new one.")
        with open(pdf_file_path, "rb") as f:
            sha3_256 = hashlib.sha3_256()
            for byte_block in iter(lambda: f.read(4096), b""):
                sha3_256.update(byte_block)
        file_hash = sha3_256.hexdigest()
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                     (file_hash text PRIMARY KEY, original_embeddings blob, corrected_embeddings blob)''')
        c.execute("SELECT * FROM embeddings WHERE file_hash=?", (file_hash,))
        row = c.fetchone()
        if row:
            original_embeddings = pickle.loads(row[1])
            corrected_embeddings = pickle.loads(row[2])

    def calculate_and_store_embeddings(sentences, embeddings):
        tokens = 0
        if embeddings is None:
            embeddings = {}
            for sentence in sentences:
                response = client.embeddings.create(model="text-embedding-ada-002", input=[sentence])
                tokens += response.usage.total_tokens
                embeddings[sentence] = response.data[0].embedding
        return embeddings, tokens

    original_sentences = [s.strip() for s in raw_text.split('. ') if s.strip()]
    corrected_sentences = [s.strip() for s in corrected_text.split('. ') if s.strip()]
    
    original_embeddings, tokens = calculate_and_store_embeddings(original_sentences, original_embeddings)
    total_tokens += tokens
    corrected_embeddings, tokens = calculate_and_store_embeddings(corrected_sentences, corrected_embeddings)
    total_tokens += tokens

    # Save the embeddings to the database
    if db_path is not None and pdf_file_path is not None:
        c.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                  (file_hash, pickle.dumps(original_embeddings), pickle.dumps(corrected_embeddings)))
        conn.commit()

    original_length = len(raw_text)
    filtered_corrected_text = corrected_text
    last_threshold = threshold
    last_filtered_text = filtered_corrected_text

    while True:
        filtered_sentences = []
        for corrected_sentence in corrected_sentences:
            corrected_embedding = corrected_embeddings.get(corrected_sentence)
            if corrected_embedding is None:
                continue
            similarities = [cosine_similarity(corrected_embedding, original_embeddings.get(original_sentence, [0]*512)) for original_sentence in original_sentences]
            max_similarity = max(similarities) if similarities else 0
            if max_similarity >= threshold:
                filtered_sentences.append(corrected_sentence)

        filtered_corrected_text = ". ".join(filtered_sentences)
        if len(filtered_corrected_text) < original_length - 30:
            filtered_corrected_text = last_filtered_text
            threshold = last_threshold
            break
        else:
            last_threshold = threshold
            last_filtered_text = filtered_corrected_text
            threshold += threshold_increment

    return filtered_corrected_text, original_embeddings, corrected_embeddings, total_tokens

def tesseract_with_llm_correction(input_pdf_file_path, 
                                  max_test_pages=0,
                                  skip_first_n_pages=0, 
                                  starting_hallucination_similarity_threshold=0.30, 
                                  check_if_valid_english=False, reformat_as_markdown=True, 
                                  sentence_embeddings_db_path="./sentence_embeddings.sqlite", 
                                  test_filtering_hallucinations=False):
    
    if not test_filtering_hallucinations:
        list_of_scanned_images = convert_pdf_to_images_func(input_pdf_file_path, max_test_pages)
        print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
        print("Extracting text from converted pages...")
        
        performance_metrics = [
            {
            'Execution Time (seconds)': 0,
            'CPU Usage (percent)': 0,
            'Memory Usage (MB)': 0,
            'llm_tokens': 0,
            'embedding_tokens': 0,
            'pages_calls': 0
            } for _ in range(3)
        ]
        
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

        def process_text(ii_text_tuple):
            ii, text = ii_text_tuple
            if ii < skip_first_n_pages:
                return None
            extracted_text_string = check_extracted_pages_func(text)
            if extracted_text_string:
                print(f"Processing page {ii + 1} with LLM...")
                corrected_extracted_text_string, llm_tokens = process_text_with_llm(extracted_text_string, check_if_valid_english, reformat_as_markdown)
                return corrected_extracted_text_string, llm_tokens
            return None

        with ThreadPoolExecutor() as executor:
            list_of_extracted_text_strings = list(executor.map(ocr_image, list_of_scanned_images))
        raw_ocr_output = "\n".join(list_of_extracted_text_strings)
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

        performance_metrics[0]['Execution Time (seconds)'] = end_time - start_time
        performance_metrics[0]['CPU Usage (percent)'] = end_cpu - start_cpu
        performance_metrics[0]['Memory Usage (MB)'] = end_memory - start_memory
        performance_metrics[0]['llm_tokens'] = 0
        performance_metrics[0]['embedding_tokens'] = 0
        performance_metrics[0]['pages_calls'] = 0

        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

        # process the OCR output
        with ThreadPoolExecutor() as executor:
            results = list(filter(None, executor.map(process_text, enumerate(list_of_extracted_text_strings))))
        # separate the corrected text strings and llm tokens
        list_of_corrected_text_strings, llm_tokens_list = zip(*results)
        # join the list of strings into a single string with a newline after each page
        final_text = "\n".join(list_of_corrected_text_strings)
        # sum up all the llm tokens
        total_llm_tokens = sum(llm_tokens_list)

        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

        performance_metrics[1]['Execution Time (seconds)'] = end_time - start_time + performance_metrics[0]['Execution Time (seconds)']
        performance_metrics[1]['CPU Usage (percent)'] = end_cpu - start_cpu + performance_metrics[0]['CPU Usage (percent)']
        performance_metrics[1]['Memory Usage (MB)'] = end_memory - start_memory + performance_metrics[0]['Memory Usage (MB)']
        performance_metrics[1]['llm_tokens'] = total_llm_tokens + performance_metrics[0]['llm_tokens']
        performance_metrics[1]['embedding_tokens'] = 0 + performance_metrics[0]['embedding_tokens']
        performance_metrics[1]['pages_calls'] = 0 + performance_metrics[0]['pages_calls']

 
    if test_filtering_hallucinations: #For debugging
        base_name = os.path.splitext(input_pdf_file_path)[0]
        output_extension = '.md' if reformat_as_markdown else '.txt'    
        output_file_path = base_name + output_extension
        base_name = os.path.splitext(input_pdf_file_path)[0]
        raw_ocr_output_file_path = f"{base_name}__raw_ocr_output.txt"        
        with open(output_file_path, 'r') as f:
            final_text = f.read()
        with open(raw_ocr_output_file_path, 'r') as f:
            raw_ocr_output = f.read()
    
    print('Now filtering out hallucinations from corrected text...')
    
    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=None)
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    filtered_output, original_embeddings, corrected_embeddings, embedding_tokens = filter_hallucinations(final_text, raw_ocr_output, starting_hallucination_similarity_threshold, input_pdf_file_path, sentence_embeddings_db_path)
    
    end_time = time.time()
    end_cpu = psutil.cpu_percent(interval=None)
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    performance_metrics[2]['Execution Time (seconds)'] = end_time - start_time + performance_metrics[1]['Execution Time (seconds)']
    performance_metrics[2]['CPU Usage (percent)'] = end_cpu - start_cpu + performance_metrics[1]['CPU Usage (percent)']
    performance_metrics[2]['Memory Usage (MB)'] = end_memory - start_memory + performance_metrics[1]['Memory Usage (MB)']
    performance_metrics[2]['llm_tokens'] = 0 + performance_metrics[1]['llm_tokens']
    performance_metrics[2]['embedding_tokens'] = embedding_tokens + performance_metrics[1]['embedding_tokens']
    performance_metrics[2]['pages_calls'] = 0 + performance_metrics[1]['pages_calls']
    
    print('Done filtering out hallucinations.')

    return raw_ocr_output, final_text, filtered_output, performance_metrics


if __name__ == '__main__':
    input_pdf_file_path = '160301289-Warren-Buffett-Katharine-Graham-Letter.pdf'
    max_test_pages = 2 # set to 0 to convert all pages of the PDF file using Tesseract
    skip_first_n_pages = 0 # set to 0 to process all pages with the LLM
    starting_hallucination_similarity_threshold = 0.30 # The higher you set this, the more potential hallucinations will be filtered out (but also the more potential correct sentences will be filtered out)
    check_if_valid_english = False # set to True to check if the extracted text is valid English
    reformat_as_markdown = True # set to True to reformat the corrected extracted text using markdown formatting
    sentence_embeddings_db_path = "./sentence_embeddings.sqlite"
    test_filtering_hallucinations = False # set to True to test filtering hallucinations (for debugging purposes)

    raw_ocr_output, final_text, filtered_output, performance_metrics  =  tesseract_with_llm_correction(input_pdf_file_path, 
                                  max_test_pages, 
                                  skip_first_n_pages, 
                                  starting_hallucination_similarity_threshold, 
                                  check_if_valid_english, reformat_as_markdown, 
                                  sentence_embeddings_db_path, 
                                  test_filtering_hallucinations)
    
    print("Performance Metrics:")
    for i, metrics in enumerate(performance_metrics):
        print(f"Page {i + 1}:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    base_name = os.path.splitext(input_pdf_file_path)[0]
    
    raw_ocr_output_file_path = f"{base_name}__raw_ocr_output.txt"
    with open(raw_ocr_output_file_path, "w") as f:
        f.write(raw_ocr_output)

    output_extension = '.md' if reformat_as_markdown else '.txt'
    # create the output file path
    output_file_path = base_name + output_extension
    
    print(f"LLM corrected text written to: {output_file_path}")
    with open(output_file_path, 'w') as f:
            f.write(final_text)

    final_output_file_path = base_name + '_filtered' + output_extension
    with open(final_output_file_path, 'w') as f:
        f.write(filtered_output)
