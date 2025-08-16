"""
Dataset Loading and Processing for RC-Mamba Research.

This module provides utilities for loading and preprocessing various datasets
used in the research evaluation including:
- Long-context datasets (NarrativeQA, Qasper)
- Multimodal datasets (VQAv2, COCO, Flickr30k) 
- Cross-lingual datasets (XNLI, Multi30k)
- Audio datasets (LibriSpeech, Common Voice)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json
import requests
from datasets import load_dataset
from PIL import Image
import librosa
from transformers import AutoTokenizer
import io
import base64
import warnings


class NarrativeQADataset(Dataset):
    """Dataset for long-context reading comprehension."""
    
    def __init__(
        self,
        split: str = "validation",
        max_samples: int = 1000,
        max_context_length: int = 8192,
        tokenizer: Optional[AutoTokenizer] = None
    ):
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer
        
        try:
            # Load NarrativeQA dataset
            self.dataset = load_dataset("narrativeqa", split=split)
            if max_samples and len(self.dataset) > max_samples:
                indices = np.random.choice(len(self.dataset), max_samples, replace=False)
                self.dataset = self.dataset.select(indices)
        except Exception as e:
            warnings.warn(f"Could not load NarrativeQA: {e}. Using synthetic data.")
            self.dataset = self._create_synthetic_long_context_data(max_samples)
    
    def _create_synthetic_long_context_data(self, num_samples: int) -> List[Dict]:
        """Create synthetic long-context data for testing."""
        synthetic_data = []
        
        base_text = """
        In the ancient kingdom of Aetheria, there lived a young scholar named Elena who spent her days in the vast royal library. 
        The library contained thousands of scrolls and books, each holding secrets of magic, history, and forgotten lore. 
        Elena was particularly fascinated by the chronicles of the Dragon Wars, a series of conflicts that had shaped the realm centuries ago.
        """
        
        for i in range(num_samples):
            # Create long context by repeating and varying the base text
            context_parts = []
            for j in range(20):  # Create ~20 paragraphs
                varied_text = base_text.replace("Elena", f"character_{j%5}")
                varied_text = varied_text.replace("Aetheria", f"Kingdom_{j%3}")
                context_parts.append(varied_text)
            
            # Insert key information at random position
            key_info = f"The secret number mentioned in scroll {i} was {1000 + i}."
            insert_pos = np.random.randint(5, 15)
            context_parts.insert(insert_pos, key_info)
            
            full_context = " ".join(context_parts)
            question = f"What was the secret number mentioned in scroll {i}?"
            answer = str(1000 + i)
            
            synthetic_data.append({
                "document": {"text": full_context},
                "question": {"text": question},
                "answers": [{"text": answer}],
                "id": f"synthetic_{i}"
            })
        
        return synthetic_data
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract fields
        if isinstance(item.get("document"), dict):
            context = item["document"]["text"]
        else:
            context = str(item.get("document", ""))
        
        if isinstance(item.get("question"), dict):
            question = item["question"]["text"]
        else:
            question = str(item.get("question", ""))
        
        # Handle answers
        answers = item.get("answers", [])
        if answers and isinstance(answers[0], dict):
            answer = answers[0]["text"]
        elif answers:
            answer = str(answers[0])
        else:
            answer = ""
        
        # Truncate context if needed
        if self.tokenizer and len(self.tokenizer.encode(context)) > self.max_context_length:
            tokens = self.tokenizer.encode(context)[:self.max_context_length]
            context = self.tokenizer.decode(tokens)
        
        return {
            "context": context,
            "question": question,
            "answer": answer,
            "id": item.get("id", f"item_{idx}"),
            "modality": "text"
        }


class VQADataset(Dataset):
    """Visual Question Answering dataset."""
    
    def __init__(
        self,
        split: str = "validation",
        max_samples: int = 1000,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.image_size = image_size
        
        try:
            # Load VQA dataset
            self.dataset = load_dataset("HuggingFaceM4/VQAv2", split=split)
            if max_samples and len(self.dataset) > max_samples:
                indices = np.random.choice(len(self.dataset), max_samples, replace=False)
                self.dataset = self.dataset.select(indices)
        except Exception as e:
            warnings.warn(f"Could not load VQA: {e}. Using synthetic data.")
            self.dataset = self._create_synthetic_vqa_data(max_samples)
    
    def _create_synthetic_vqa_data(self, num_samples: int) -> List[Dict]:
        """Create synthetic VQA data for testing."""
        questions = [
            "What color is the main object?",
            "How many items are visible?", 
            "What is the weather like?",
            "Where is this scene taking place?",
            "What animals can you see?",
            "What time of day is it?",
            "What is the person doing?",
            "What objects are on the table?",
            "What is the dominant color?",
            "Is this indoors or outdoors?"
        ]
        
        answers = [
            "blue", "three", "sunny", "park", "dog",
            "afternoon", "reading", "book, cup", "green", "outdoors"
        ]
        
        synthetic_data = []
        for i in range(num_samples):
            # Create a dummy image (solid color)
            color = np.random.randint(0, 255, 3)
            image_array = np.full((224, 224, 3), color, dtype=np.uint8)
            image = Image.fromarray(image_array)
            
            question = questions[i % len(questions)]
            answer = answers[i % len(answers)]
            
            synthetic_data.append({
                "image": image,
                "question": question,
                "answer": answer,
                "image_id": i,
                "question_id": i
            })
        
        return synthetic_data
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Handle image
        image = item.get("image")
        if image is None:
            # Create placeholder image
            image = Image.new('RGB', self.image_size, color='gray')
        elif hasattr(image, 'resize'):
            image = image.resize(self.image_size)
        
        return {
            "image": image,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "image_id": item.get("image_id", idx),
            "question_id": item.get("question_id", idx),
            "modality": "vision"
        }


class XNLIDataset(Dataset):
    """Cross-lingual Natural Language Inference dataset."""
    
    def __init__(
        self,
        split: str = "validation",
        languages: List[str] = None,
        max_samples: int = 1000
    ):
        if languages is None:
            languages = ["en", "fr", "de", "es", "zh", "ja", "ar"]
        
        self.languages = languages
        
        try:
            # Load XNLI dataset
            self.dataset = load_dataset("xnli", split=split)
            
            # Filter by languages
            self.dataset = self.dataset.filter(lambda x: x["language"] in languages)
            
            if max_samples and len(self.dataset) > max_samples:
                indices = np.random.choice(len(self.dataset), max_samples, replace=False)
                self.dataset = self.dataset.select(indices)
        except Exception as e:
            warnings.warn(f"Could not load XNLI: {e}. Using synthetic data.")
            self.dataset = self._create_synthetic_xnli_data(max_samples)
    
    def _create_synthetic_xnli_data(self, num_samples: int) -> List[Dict]:
        """Create synthetic XNLI data for testing."""
        premises = {
            "en": ["The cat is sleeping.", "It's raining today.", "Students are studying."],
            "fr": ["Le chat dort.", "Il pleut aujourd'hui.", "Les étudiants étudient."],
            "de": ["Die Katze schläft.", "Es regnet heute.", "Studenten lernen."],
            "es": ["El gato está durmiendo.", "Está lloviendo hoy.", "Los estudiantes estudian."],
        }
        
        hypotheses = {
            "en": ["The cat is awake.", "The weather is nice.", "People are reading."],
            "fr": ["Le chat est éveillé.", "Le temps est beau.", "Les gens lisent."],
            "de": ["Die Katze ist wach.", "Das Wetter ist schön.", "Leute lesen."],
            "es": ["El gato está despierto.", "El clima es agradable.", "La gente lee."],
        }
        
        labels = [2, 2, 1]  # contradiction, contradiction, entailment
        
        synthetic_data = []
        for i in range(num_samples):
            lang = self.languages[i % len(self.languages)]
            if lang not in premises:
                lang = "en"  # fallback
            
            premise_idx = i % len(premises[lang])
            premise = premises[lang][premise_idx]
            hypothesis = hypotheses[lang][premise_idx]
            label = labels[premise_idx]
            
            synthetic_data.append({
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "language": lang
            })
        
        return synthetic_data
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        return {
            "premise": item.get("premise", ""),
            "hypothesis": item.get("hypothesis", ""),
            "label": item.get("label", 0),
            "language": item.get("language", "en"),
            "modality": "text"
        }


class LibriSpeechDataset(Dataset):
    """Speech recognition dataset."""
    
    def __init__(
        self,
        split: str = "validation.clean",
        max_samples: int = 500,
        sample_rate: int = 16000,
        max_duration: float = 20.0
    ):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        
        try:
            # Load LibriSpeech dataset
            self.dataset = load_dataset("librispeech_asr", split=split, streaming=False)
            if max_samples and len(self.dataset) > max_samples:
                indices = np.random.choice(len(self.dataset), max_samples, replace=False)
                self.dataset = self.dataset.select(indices)
        except Exception as e:
            warnings.warn(f"Could not load LibriSpeech: {e}. Using synthetic data.")
            self.dataset = self._create_synthetic_audio_data(max_samples)
    
    def _create_synthetic_audio_data(self, num_samples: int) -> List[Dict]:
        """Create synthetic audio data for testing."""
        texts = [
            "Hello, how are you today?",
            "The weather is quite nice.",
            "Machine learning is fascinating.",
            "Please speak clearly and slowly.",
            "Thank you for your attention."
        ]
        
        synthetic_data = []
        for i in range(num_samples):
            # Generate synthetic audio (simple sine wave)
            duration = np.random.uniform(1.0, 5.0)
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            frequency = 440 + np.random.uniform(-100, 100)  # Around A4
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            text = texts[i % len(texts)]
            
            synthetic_data.append({
                "audio": {"array": audio, "sampling_rate": self.sample_rate},
                "text": text,
                "id": f"synthetic_{i}"
            })
        
        return synthetic_data
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract audio
        audio_data = item.get("audio", {})
        if isinstance(audio_data, dict):
            audio_array = audio_data.get("array", np.array([]))
            sample_rate = audio_data.get("sampling_rate", self.sample_rate)
        else:
            audio_array = np.array([])
            sample_rate = self.sample_rate
        
        # Resample if necessary
        if sample_rate != self.sample_rate and len(audio_array) > 0:
            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=self.sample_rate
            )
        
        # Truncate if too long
        max_samples = int(self.max_duration * self.sample_rate)
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]
        
        return {
            "audio": audio_array,
            "text": item.get("text", ""),
            "sample_rate": self.sample_rate,
            "id": item.get("id", f"item_{idx}"),
            "modality": "audio"
        }


class MultimodalDataset(Dataset):
    """Combined multimodal dataset."""
    
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        sampling_weights: Optional[Dict[str, float]] = None
    ):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        
        # Calculate total length
        self.lengths = {name: len(dataset) for name, dataset in datasets.items()}
        self.total_length = sum(self.lengths.values())
        
        # Set sampling weights
        if sampling_weights is None:
            self.sampling_weights = {name: 1.0 for name in self.dataset_names}
        else:
            self.sampling_weights = sampling_weights
        
        # Normalize weights
        total_weight = sum(self.sampling_weights.values())
        self.sampling_weights = {
            name: weight / total_weight 
            for name, weight in self.sampling_weights.items()
        }
        
        # Create sampling indices
        self._create_sampling_indices()
    
    def _create_sampling_indices(self):
        """Create indices for weighted sampling."""
        self.indices = []
        
        for name, weight in self.sampling_weights.items():
            dataset_size = self.lengths[name]
            num_samples = int(weight * self.total_length)
            
            # Sample with replacement if needed
            if num_samples > dataset_size:
                dataset_indices = np.random.choice(dataset_size, num_samples, replace=True)
            else:
                dataset_indices = np.random.choice(dataset_size, num_samples, replace=False)
            
            for idx in dataset_indices:
                self.indices.append((name, idx))
        
        # Shuffle the combined indices
        np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        dataset_name, dataset_idx = self.indices[idx]
        item = self.datasets[dataset_name][dataset_idx]
        
        # Add dataset information
        item["dataset"] = dataset_name
        
        return item


class DatasetFactory:
    """Factory for creating datasets used in RC-Mamba research."""
    
    @staticmethod
    def create_long_context_dataset(
        max_samples: int = 1000,
        max_context_length: int = 8192,
        tokenizer: Optional[AutoTokenizer] = None
    ) -> NarrativeQADataset:
        """Create long-context dataset."""
        return NarrativeQADataset(
            max_samples=max_samples,
            max_context_length=max_context_length,
            tokenizer=tokenizer
        )
    
    @staticmethod
    def create_multimodal_dataset(
        max_samples_per_modality: int = 500
    ) -> MultimodalDataset:
        """Create combined multimodal dataset."""
        datasets = {
            "vqa": VQADataset(max_samples=max_samples_per_modality),
            "xnli": XNLIDataset(max_samples=max_samples_per_modality),
            "librispeech": LibriSpeechDataset(max_samples=max_samples_per_modality // 2)
        }
        
        return MultimodalDataset(
            datasets=datasets,
            sampling_weights={"vqa": 0.4, "xnli": 0.4, "librispeech": 0.2}
        )
    
    @staticmethod
    def create_evaluation_datasets(
        tokenizer: Optional[AutoTokenizer] = None
    ) -> Dict[str, Dataset]:
        """Create all evaluation datasets."""
        return {
            "long_context": DatasetFactory.create_long_context_dataset(
                max_samples=500, tokenizer=tokenizer
            ),
            "vqa": VQADataset(max_samples=300),
            "xnli": XNLIDataset(max_samples=300),
            "librispeech": LibriSpeechDataset(max_samples=100),
            "multimodal": DatasetFactory.create_multimodal_dataset(max_samples_per_modality=200)
        }


def create_research_dataloaders(
    batch_size: int = 8,
    num_workers: int = 4,
    tokenizer: Optional[AutoTokenizer] = None
) -> Dict[str, DataLoader]:
    """Create data loaders for research evaluation."""
    
    datasets = DatasetFactory.create_evaluation_datasets(tokenizer)
    
    dataloaders = {}
    for name, dataset in datasets.items():
        dataloaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Evaluation should be deterministic
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=lambda batch: batch  # Simple collation
        )
    
    return dataloaders


# Example usage and testing
if __name__ == "__main__":
    print("Testing dataset loading...")
    
    # Test individual datasets
    print("\n1. Testing NarrativeQA dataset...")
    narrativeqa = NarrativeQADataset(max_samples=5)
    sample = narrativeqa[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Context length: {len(sample['context'])}")
    print(f"Question: {sample['question'][:100]}...")
    print(f"Answer: {sample['answer']}")
    
    print("\n2. Testing VQA dataset...")
    vqa = VQADataset(max_samples=5)
    sample = vqa[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print(f"Image size: {sample['image'].size}")
    
    print("\n3. Testing XNLI dataset...")
    xnli = XNLIDataset(max_samples=5)
    sample = xnli[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Language: {sample['language']}")
    print(f"Premise: {sample['premise']}")
    print(f"Hypothesis: {sample['hypothesis']}")
    print(f"Label: {sample['label']}")
    
    print("\n4. Testing LibriSpeech dataset...")
    librispeech = LibriSpeechDataset(max_samples=3)
    sample = librispeech[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Audio shape: {sample['audio'].shape}")
    print(f"Text: {sample['text']}")
    
    print("\n5. Testing multimodal dataset...")
    multimodal = DatasetFactory.create_multimodal_dataset(max_samples_per_modality=10)
    print(f"Total length: {len(multimodal)}")
    
    sample = multimodal[0]
    print(f"Sample modality: {sample['modality']}")
    print(f"Dataset: {sample['dataset']}")
    
    print("\nDataset testing completed successfully!")
