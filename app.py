import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import os
from typing import Tuple
import torch.nn.functional as F
import pymupdf4llm
import tempfile

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF using PyMuPDF"""
    # Save uploaded file to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Initialize parser
        text = pymupdf4llm.to_markdown(tmp_path)
        return text
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

def generate(model, input_ids: torch.Tensor, past_key_values, max_new_tokens: int = 50) -> torch.Tensor:
    device = model.model.embed_tokens.weight.device
    origin_len = input_ids.shape[-1]
    input_ids = input_ids.to(device)
    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = out.logits[:, -1, :]
            token = torch.argmax(logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, token], dim=-1)
            past_key_values = out.past_key_values
            next_token = token.to(device)

            if model.config.eos_token_id is not None and token.item() == model.config.eos_token_id:
                break
    return output_ids[:, origin_len:]

def get_kv_cache(model, tokenizer, prompt: str) -> DynamicCache:
    device = model.model.embed_tokens.weight.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cache = DynamicCache()

    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True
        )
    return cache

def clean_up(cache: DynamicCache, origin_len: int):
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, :origin_len, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, :origin_len, :]

class CacheAugmentedMistral:
    def __init__(self, model_name="Qwen/Qwen2-0.5B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")
        )

    def initialize_cache(self, context: str) -> Tuple[DynamicCache, int]:
        system_prompt = f"""
        <|system|>
        You are an assistant who provides concise factual answers.
        <|user|>
        Context:
        {context}
        Question:
        """.strip()
        
        cache = get_kv_cache(self.model, self.tokenizer, system_prompt)
        origin_len = cache.key_cache[0].shape[-2]
        return cache, origin_len

    def generate_response(self, question: str, cache: DynamicCache, origin_len: int) -> str:
        clean_up(cache, origin_len)
        input_ids = self.tokenizer(question + "\n", return_tensors="pt").input_ids
        gen_ids = generate(self.model, input_ids, cache)
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

def main():
    st.title("PDF Question-Answering with Cache-Augmented Mistral")
    
    # Initialize the system
    if 'mistral' not in st.session_state:
        with st.spinner("Loading Mistral model..."):
            st.session_state.mistral = CacheAugmentedMistral()
            st.success("Model loaded successfully!")

    # Initialize session state
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
        st.session_state.cache = None
        st.session_state.origin_len = None

    # PDF upload
    uploaded_file = st.file_uploader("Upload PDF document", type=['pdf'])
    
    if uploaded_file is not None:
        # Extract text from PDF
        with st.spinner("Processing PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
            
            # Only update if text content has changed
            if pdf_text != st.session_state.pdf_text:
                st.session_state.pdf_text = pdf_text
                
                # Show extracted text in expandable section
                with st.expander("View Extracted Text"):
                    st.text_area("PDF Content", pdf_text, height=200, disabled=True)
                
                # Build cache with new content
                with st.spinner("Building KV cache..."):
                    st.session_state.cache, st.session_state.origin_len = (
                        st.session_state.mistral.initialize_cache(pdf_text)
                    )
                st.success("PDF processed and KV cache built successfully!")

    # Question input and generation
    if st.session_state.cache is not None:
        question = st.text_input("Enter your question about the PDF:")
        
        if st.button("Generate Answer") and question:
            with st.spinner("Generating response..."):
                response = st.session_state.mistral.generate_response(
                    question,
                    st.session_state.cache,
                    st.session_state.origin_len
                )
            
            st.subheader("Response:")
            st.write(response)
    else:
        st.info("Please upload a PDF document to start asking questions.")

if __name__ == "__main__":
    main()