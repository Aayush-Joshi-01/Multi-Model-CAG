import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import os
from typing import Tuple, Dict, Any
import torch.nn.functional as F
import pymupdf4llm
import tempfile
from abc import ABC, abstractmethod

class DocumentProcessor(ABC):
    """Abstract base class for document processors"""
    @abstractmethod
    def extract_text(self, file) -> str:
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> list:
        pass

class PDFProcessor(DocumentProcessor):
    def extract_text(self, file) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        try:
            text = pymupdf4llm.to_markdown(tmp_path)
            return text
        finally:
            os.unlink(tmp_path)
            
    def get_supported_extensions(self) -> list:
        return ['pdf']

class TXTProcessor(DocumentProcessor):
    def extract_text(self, file) -> str:
        return file.getvalue().decode('utf-8')
    
    def get_supported_extensions(self) -> list:
        return ['txt']

class ModelConfig:
    """Configuration class for different models"""
    def __init__(
        self,
        name: str,
        model_id: str,
        system_prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7
    ):
        self.name = name
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

def generate(
    model,
    input_ids: torch.Tensor,
    past_key_values,
    max_new_tokens: int = 50,
    temperature: float = 0.7
) -> torch.Tensor:
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
            # Apply temperature
            if temperature != 0:
                logits = logits / temperature
            token = torch.argmax(logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, token], dim=-1)
            past_key_values = out.past_key_values
            next_token = token.to(device)

            if model.config.eos_token_id is not None and token.item() == model.config.eos_token_id:
                break
    return output_ids[:, origin_len:]

class CacheAugmentedQA:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_id,
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")
        )

    def initialize_cache(self, context: str) -> Tuple[DynamicCache, int]:
        formatted_prompt = self.model_config.system_prompt.format(context=context)
        cache = self._get_kv_cache(formatted_prompt)
        origin_len = cache.key_cache[0].shape[-2]
        return cache, origin_len

    def _get_kv_cache(self, prompt: str) -> DynamicCache:
        device = self.model.model.embed_tokens.weight.device
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        cache = DynamicCache()

        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True
            )
        return cache

    def _clean_up_cache(self, cache: DynamicCache, origin_len: int):
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][:, :, :origin_len, :]
            cache.value_cache[i] = cache.value_cache[i][:, :, :origin_len, :]

    def generate_response(self, question: str, cache: DynamicCache, origin_len: int) -> str:
        self._clean_up_cache(cache, origin_len)
        input_ids = self.tokenizer(question + "\n", return_tensors="pt").input_ids
        gen_ids = generate(
            self.model,
            input_ids,
            cache,
            max_new_tokens=self.model_config.max_new_tokens,
            temperature=self.model_config.temperature
        )
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

def get_available_models() -> Dict[str, ModelConfig]:
    return {
        "Qwen 0.5B": ModelConfig(
            name="Qwen 0.5B",
            model_id="Qwen/Qwen2-0.5B-Instruct",
            system_prompt="""
            <|system|>
            You are an assistant who provides concise factual answers.
            <|user|>
            Context:
            {context}
            Question:
            """.strip()
        ),
        "Mistral 7B": ModelConfig(
            name="Mistral 7B",
            model_id="mistralai/Mistral-7B-Instruct-v0.1",
            system_prompt="""
            <s>[INST] You are an assistant who provides concise factual answers.
            Context:
            {context}
            Question: [/INST]
            """.strip()
        ),
        "Qwen 2B": ModelConfig(
            name="Qwen 2B",
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            system_prompt="""
            <|system|>
            You are an assistant who provides concise factual answers.
            <|user|>
            Context:
            {context}
            Question:
            """.strip()
        )
    }

def get_document_processors() -> Dict[str, DocumentProcessor]:
    return {
        "PDF": PDFProcessor(),
        "TXT": TXTProcessor()
    }

def main():
    st.title("Multi-Model Document Question-Answering System")
    
    available_models = get_available_models()
    document_processors = get_document_processors()
    
    initialize_session_state()
    
    selected_model = model_selection(available_models)
    
    if model_needs_initialization(selected_model):
        initialize_model(selected_model, available_models)
    
    supported_extensions = get_supported_extensions(document_processors)
    
    uploaded_file = st.file_uploader("Upload document", type=supported_extensions)
    
    if uploaded_file is not None:
        process_uploaded_file(uploaded_file, document_processors)
    
    if st.session_state.cache is not None:
        handle_question_input()
    else:
        st.info("Please upload a document to start asking questions.")

def initialize_session_state():
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = None
        st.session_state.doc_text = ""
        st.session_state.cache = None
        st.session_state.origin_len = None

def model_selection(available_models):
    return st.selectbox(
        "Select Model",
        options=list(available_models.keys()),
        key="model_selection"
    )

def model_needs_initialization(selected_model):
    return (st.session_state.qa_system is None or 
            st.session_state.qa_system.model_config.name != selected_model)

def initialize_model(selected_model, available_models):
    with st.spinner(f"Loading {selected_model} model..."):
        st.session_state.qa_system = CacheAugmentedQA(available_models[selected_model])
        st.success("Model loaded successfully!")

def get_supported_extensions(document_processors):
    supported_extensions = []
    for processor in document_processors.values():
        supported_extensions.extend(processor.get_supported_extensions())
    return supported_extensions

def process_uploaded_file(uploaded_file, document_processors):
    file_ext = uploaded_file.name.split('.')[-1].lower()
    processor = get_processor_for_extension(file_ext, document_processors)
    
    if processor:
        with st.spinner("Processing document..."):
            doc_text = processor.extract_text(uploaded_file)
            
            if doc_text != st.session_state.doc_text:
                st.session_state.doc_text = doc_text
                
                with st.expander("View Extracted Text"):
                    st.text_area("Document Content", doc_text, height=200, disabled=True)
                
                with st.spinner("Building KV cache..."):
                    st.session_state.cache, st.session_state.origin_len = (
                        st.session_state.qa_system.initialize_cache(doc_text)
                    )
                st.success("Document processed and KV cache built successfully!")

def get_processor_for_extension(file_ext, document_processors):
    for p in document_processors.values():
        if file_ext in p.get_supported_extensions():
            return p
    return None

def handle_question_input():
    question = st.text_input("Enter your question about the document:")
    
    if st.button("Generate Answer") and question:
        with st.spinner("Generating response..."):
            response = st.session_state.qa_system.generate_response(
                question,
                st.session_state.cache,
                st.session_state.origin_len
            )
        
        st.subheader("Response:")
        st.write(response)

if __name__ == "__main__":
    main()