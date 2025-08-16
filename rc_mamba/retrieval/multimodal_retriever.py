"""
Cross-Modal and Cross-Lingual Retrieval System for RC-Mamba.

This module implements a unified retrieval pipeline that can handle text, image,
and audio inputs across multiple languages. It uses state-of-the-art encoders
and provides a seamless interface for multimodal retrieval.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Dict, Optional, Tuple, Any
from transformers import (
    AutoTokenizer, AutoModel,
    CLIPProcessor, CLIPModel,
    Wav2Vec2Processor, Wav2Vec2Model
)
import faiss
from sentence_transformers import SentenceTransformer
import librosa
from PIL import Image


class MultiModalEncoder(nn.Module):
    """Unified encoder for text, image, and audio modalities."""
    
    def __init__(
        self,
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vision_model: str = "openai/clip-vit-base-patch32",
        audio_model: str = "facebook/wav2vec2-base-960h",
        embedding_dim: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        
        # Text encoder (multilingual)
        self.text_encoder = SentenceTransformer(text_model, device=device)
        
        # Vision encoder
        self.vision_model = CLIPModel.from_pretrained(vision_model).to(device)
        self.vision_processor = CLIPProcessor.from_pretrained(vision_model)
        
        # Audio encoder
        self.audio_model = Wav2Vec2Model.from_pretrained(audio_model).to(device)
        self.audio_processor = Wav2Vec2Processor.from_pretrained(audio_model)
        
        # Projection layers to unified embedding space
        text_dim = self.text_encoder.get_sentence_embedding_dimension()
        vision_dim = self.vision_model.config.projection_dim
        audio_dim = self.audio_model.config.hidden_size
        
        self.text_proj = nn.Linear(text_dim, embedding_dim).to(device)
        self.vision_proj = nn.Linear(vision_dim, embedding_dim).to(device)
        self.audio_proj = nn.Linear(audio_dim, embedding_dim).to(device)
        
        # Layer normalization for stability
        self.text_norm = nn.LayerNorm(embedding_dim).to(device)
        self.vision_norm = nn.LayerNorm(embedding_dim).to(device)
        self.audio_norm = nn.LayerNorm(embedding_dim).to(device)
        
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text(s) to unified embedding space."""
        if isinstance(texts, str):
            texts = [texts]
        
        with torch.no_grad():
            embeddings = self.text_encoder.encode(texts, convert_to_tensor=True)
            embeddings = self.text_proj(embeddings)
            embeddings = self.text_norm(embeddings)
        
        return embeddings
    
    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Encode image(s) to unified embedding space."""
        if isinstance(images, Image.Image):
            images = [images]
        
        with torch.no_grad():
            inputs = self.vision_processor(images=images, return_tensors="pt").to(self.device)
            image_features = self.vision_model.get_image_features(**inputs)
            embeddings = self.vision_proj(image_features)
            embeddings = self.vision_norm(embeddings)
        
        return embeddings
    
    def encode_audio(self, audio_arrays: Union[np.ndarray, List[np.ndarray]], 
                    sample_rate: int = 16000) -> torch.Tensor:
        """Encode audio to unified embedding space."""
        if isinstance(audio_arrays, np.ndarray):
            audio_arrays = [audio_arrays]
        
        with torch.no_grad():
            # Process audio inputs
            inputs = self.audio_processor(
                audio_arrays, 
                sampling_rate=sample_rate, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            # Get hidden states and pool
            outputs = self.audio_model(**inputs)
            # Use mean pooling over sequence dimension
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = self.audio_proj(embeddings)
            embeddings = self.audio_norm(embeddings)
        
        return embeddings
    
    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        """Forward pass for mixed modality inputs."""
        embeddings = []
        
        if "text" in data:
            text_emb = self.encode_text(data["text"])
            embeddings.append(text_emb)
        
        if "image" in data:
            image_emb = self.encode_image(data["image"])
            embeddings.append(image_emb)
        
        if "audio" in data:
            audio_emb = self.encode_audio(data["audio"])
            embeddings.append(audio_emb)
        
        if embeddings:
            # Average embeddings if multiple modalities
            return torch.stack(embeddings).mean(dim=0)
        else:
            return torch.zeros(1, self.embedding_dim, device=self.device)


class CrossModalRetriever:
    """Cross-modal retrieval system with FAISS indexing."""
    
    def __init__(
        self,
        encoder: MultiModalEncoder,
        index_type: str = "IVF",
        n_clusters: int = 100,
        embedding_dim: int = 512
    ):
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        
        # Initialize FAISS index
        if index_type == "IVF":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, n_clusters)
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)
        
        self.is_trained = False
        self.corpus_data = []
        self.corpus_metadata = []
        
    def add_corpus(self, data_items: List[Dict[str, Any]], metadata: List[Dict] = None):
        """Add multimodal data items to the corpus."""
        if metadata is None:
            metadata = [{}] * len(data_items)
        
        embeddings = []
        for item in data_items:
            emb = self.encoder(item)
            embeddings.append(emb.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        
        # Train index if necessary
        if not self.is_trained and hasattr(self.index, "train"):
            self.index.train(embeddings)
            self.is_trained = True
        
        # Add to index
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Store metadata
        self.corpus_data.extend(data_items)
        self.corpus_metadata.extend(metadata)
    
    def retrieve(
        self, 
        query: Dict[str, Any], 
        k: int = 5,
        return_scores: bool = True
    ) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """Retrieve top-k similar items for a multimodal query."""
        query_embedding = self.encoder(query).cpu().numpy()
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        retrieved_items = []
        for idx in indices[0]:
            if idx < len(self.corpus_data):
                item = {
                    "data": self.corpus_data[idx],
                    "metadata": self.corpus_metadata[idx]
                }
                retrieved_items.append(item)
        
        if return_scores:
            return retrieved_items, scores[0]
        else:
            return retrieved_items, None


class MultiHopRetriever:
    """Multi-hop retrieval controller with uncertainty-based triggering."""
    
    def __init__(
        self,
        base_retriever: CrossModalRetriever,
        max_hops: int = 3,
        uncertainty_threshold: float = 2.0,
        temperature: float = 1.0
    ):
        self.base_retriever = base_retriever
        self.max_hops = max_hops
        self.uncertainty_threshold = uncertainty_threshold
        self.temperature = temperature
        
        self.current_context = None
        self.hop_history = []
        
    def reset(self):
        """Reset retrieval state for new query."""
        self.current_context = None
        self.hop_history = []
    
    def compute_uncertainty(self, logits: torch.Tensor) -> float:
        """Compute uncertainty from model logits."""
        probs = torch.softmax(logits / self.temperature, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        return entropy
    
    def should_retrieve(self, logits: torch.Tensor) -> bool:
        """Decide whether to perform another retrieval hop."""
        if len(self.hop_history) >= self.max_hops:
            return False
        
        uncertainty = self.compute_uncertainty(logits)
        return uncertainty > self.uncertainty_threshold
    
    def retrieve_hop(
        self, 
        query: Dict[str, Any], 
        context: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Perform a single retrieval hop."""
        # Augment query with context if available
        if context and "text" in query:
            query["text"] = f"{context} {query['text']}"
        
        retrieved_items, scores = self.base_retriever.retrieve(query, k=5)
        
        # Combine retrieved information
        combined_embedding = None
        retrieval_info = {
            "hop": len(self.hop_history),
            "items": retrieved_items,
            "scores": scores
        }
        
        if retrieved_items:
            # Average embeddings of top retrieved items
            embeddings = []
            for item in retrieved_items[:3]:  # Use top 3
                emb = self.base_retriever.encoder(item["data"])
                embeddings.append(emb)
            
            combined_embedding = torch.stack(embeddings).mean(dim=0)
        else:
            # Return zero embedding if no items retrieved
            combined_embedding = torch.zeros(
                1, self.base_retriever.encoder.embedding_dim,
                device=self.base_retriever.encoder.device
            )
        
        self.hop_history.append(retrieval_info)
        return combined_embedding, retrieval_info
    
    def __call__(
        self, 
        query: Dict[str, Any], 
        logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Main retrieval interface."""
        if logits is None or len(self.hop_history) == 0:
            # Initial retrieval
            embedding, _ = self.retrieve_hop(query)
            return embedding
        
        if self.should_retrieve(logits):
            # Additional hop needed
            embedding, _ = self.retrieve_hop(query, self.current_context)
            return embedding
        else:
            # Return cached embedding
            return self.hop_history[-1].get("embedding", torch.zeros(
                1, self.base_retriever.encoder.embedding_dim,
                device=self.base_retriever.encoder.device
            ))


def load_multimodal_datasets() -> Dict[str, List[Dict]]:
    """Load standard multimodal datasets for evaluation."""
    datasets = {
        "text_samples": [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "Machine learning is transforming artificial intelligence."},
            {"text": "Climate change poses significant challenges to our planet."},
        ],
        "image_samples": [
            # Placeholder - in practice, load actual images
            {"image": None, "caption": "A beautiful sunset over mountains"},
            {"image": None, "caption": "A cat sitting on a windowsill"},
            {"image": None, "caption": "Modern architecture in a city"},
        ],
        "audio_samples": [
            # Placeholder - in practice, load actual audio files
            {"audio": None, "transcript": "Hello, how are you today?"},
            {"audio": None, "transcript": "The weather is quite nice"},
            {"audio": None, "transcript": "Let me explain this concept"},
        ]
    }
    return datasets


if __name__ == "__main__":
    # Example usage
    encoder = MultiModalEncoder()
    retriever = CrossModalRetriever(encoder)
    
    # Load sample data
    datasets = load_multimodal_datasets()
    
    # Add text corpus
    retriever.add_corpus(datasets["text_samples"])
    
    # Example query
    query = {"text": "machine learning and AI"}
    results, scores = retriever.retrieve(query, k=3)
    
    print("Retrieved items:")
    for i, (item, score) in enumerate(zip(results, scores)):
        print(f"{i+1}. Score: {score:.3f}, Text: {item['data']['text']}")
