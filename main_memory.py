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
import psutil
import gc

def estimate_model_size(model_id: str) -> float:
    """
    Estimate model size in GB based on model ID
    Returns approximate memory requirements
    """
    size_map = {
        "Qwen/Qwen2-0.5B-Instruct": 1.5,  # 0.5B parameters + overhead
        "mistralai/Mistral-7B-Instruct-v0.1": 15.0,  # 7B parameters + overhead
        # Add more models and their approximate sizes
    }
    return size_map.get(model_id, 20.0)  # Default to 20GB if unknown


def get_available_memory():
    """Get available system memory in GB"""
    memory = psutil.virtual_memory()
    return memory.available / (1024 * 1024 * 1024)  # Convert to GB

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
        
        # Check available memory
        available_memory = get_available_memory()
        required_memory = estimate_model_size(model_config.model_id)
        
        if available_memory < required_memory:
            raise MemoryError(
                f"Not enough memory to load {model_config.name}. "
                f"Required: {required_memory:.1f}GB, Available: {available_memory:.1f}GB"
            )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_id,
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True
            )
            
            # Configure model loading based on available memory
            load_kwargs = {
                "trust_remote_code": True,
                "token": os.getenv("HF_TOKEN")
            }
            
            if torch.cuda.is_available():
                if model_config.load_in_4bit:
                    load_kwargs.update({
                        "load_in_4bit": True,
                        "device_map": "auto"
                    })
                elif model_config.load_in_8bit:
                    load_kwargs.update({
                        "load_in_8bit": True,
                        "device_map": "auto"
                    })
                else:
                    load_kwargs["torch_dtype"] = torch.float16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id,
                **load_kwargs
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    def __del__(self):
        """Cleanup when instance is deleted"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

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
        )
    }

def get_document_processors() -> Dict[str, DocumentProcessor]:
    return {
        "PDF": PDFProcessor(),
        "TXT": TXTProcessor()
    }

def main():
    st.title("Multi-Model Document Question-Answering System")
    display_available_memory()
    available_models = get_available_models()
    document_processors = get_document_processors()
    initialize_session_state()
    selected_model = model_selection(available_models)
    initialize_model_if_needed(selected_model, available_models)
    handle_file_upload(document_processors)
    handle_question_input()

def display_available_memory():
    available_memory = get_available_memory()
    st.sidebar.info(f"Available Memory: {available_memory:.1f}GB")

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

def initialize_model_if_needed(selected_model, available_models):
    if (st.session_state.qa_system is None or 
        st.session_state.qa_system.model_config.name != selected_model):
        try:
            with st.spinner(f"Loading {selected_model} model..."):
                if st.session_state.qa_system is not None:
                    del st.session_state.qa_system
                    torch.cuda.empty_cache()
                    gc.collect()
                st.session_state.qa_system = CacheAugmentedQA(available_models[selected_model])
                st.success("Model loaded successfully!")
        except MemoryError as e:
            st.error(str(e))
            st.error("Please select a smaller model or free up system memory.")
            return
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return

def handle_file_upload(document_processors):
    supported_extensions = []
    for processor in document_processors.values():
        supported_extensions.extend(processor.get_supported_extensions())
    uploaded_file = st.file_uploader("Upload document", type=supported_extensions)
    if uploaded_file is not None:
        process_uploaded_file(uploaded_file, document_processors)

def process_uploaded_file(uploaded_file, document_processors):
    file_ext = uploaded_file.name.split('.')[-1].lower()
    processor = None
    for p in document_processors.values():
        if file_ext in p.get_supported_extensions():
            processor = p
            break
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

def handle_question_input():
    if st.session_state.cache is not None:
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
    else:
        st.info("Please upload a document to start asking questions.")

if __name__ == "__main__":
    main()