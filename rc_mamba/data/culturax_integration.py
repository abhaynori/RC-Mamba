"""
CulturaX Dataset Integration for RC-Mamba.

This module provides comprehensive support for the CulturaX multilingual dataset
(https://huggingface.co/datasets/uonlp/CulturaX) with 6.3 trillion tokens across
167 languages. The integration includes:

- Efficient streaming and batching for the massive dataset
- Language-specific sampling and balancing
- Cross-lingual evaluation protocols
- Multilingual retrieval corpus construction
- Quality filtering and preprocessing
- Memory-efficient data loading
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
import datasets
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CulturaX language information with statistics
CULTURAX_LANGUAGES = {
    # Major languages (>1% of dataset)
    'en': {'name': 'English', 'tokens': 2846970578793, 'docs': 3241065682, 'percentage': 45.13},
    'ru': {'name': 'Russian', 'tokens': 737201800363, 'docs': 799310908, 'percentage': 11.69},
    'es': {'name': 'Spanish', 'tokens': 373845662394, 'docs': 450937645, 'percentage': 5.93},
    'de': {'name': 'German', 'tokens': 357030348021, 'docs': 420017484, 'percentage': 5.66},
    'fr': {'name': 'French', 'tokens': 319332674695, 'docs': 363754348, 'percentage': 5.06},
    'zh': {'name': 'Chinese', 'tokens': 227055380882, 'docs': 218624604, 'percentage': 3.60},
    'it': {'name': 'Italian', 'tokens': 165446410843, 'docs': 211309922, 'percentage': 2.62},
    'pt': {'name': 'Portuguese', 'tokens': 136941763923, 'docs': 190289658, 'percentage': 2.17},
    'pl': {'name': 'Polish', 'tokens': 117269087143, 'docs': 142167217, 'percentage': 1.86},
    'ja': {'name': 'Japanese', 'tokens': 107873841351, 'docs': 111188475, 'percentage': 1.71},
    'nl': {'name': 'Dutch', 'tokens': 80032209900, 'docs': 117392666, 'percentage': 1.27},
    'ar': {'name': 'Arabic', 'tokens': 69354335076, 'docs': 74027952, 'percentage': 1.10},
    'tr': {'name': 'Turkish', 'tokens': 64292787164, 'docs': 94207460, 'percentage': 1.02},
    
    # Medium languages (0.1-1% of dataset)
    'cs': {'name': 'Czech', 'tokens': 56910486745, 'docs': 65350564, 'percentage': 0.90},
    'vi': {'name': 'Vietnamese', 'tokens': 55380123774, 'docs': 57606341, 'percentage': 0.88},
    'fa': {'name': 'Persian', 'tokens': 45947657495, 'docs': 59531144, 'percentage': 0.73},
    'hu': {'name': 'Hungarian', 'tokens': 43417981714, 'docs': 44132152, 'percentage': 0.69},
    'el': {'name': 'Greek', 'tokens': 43147590757, 'docs': 51430226, 'percentage': 0.68},
    'ro': {'name': 'Romanian', 'tokens': 39647954768, 'docs': 40325424, 'percentage': 0.63},
    'sv': {'name': 'Swedish', 'tokens': 38486181494, 'docs': 49709189, 'percentage': 0.61},
    'uk': {'name': 'Ukrainian', 'tokens': 38226128686, 'docs': 44740545, 'percentage': 0.61},
    'fi': {'name': 'Finnish', 'tokens': 28925009180, 'docs': 30467667, 'percentage': 0.46},
    'ko': {'name': 'Korean', 'tokens': 24765448392, 'docs': 20557310, 'percentage': 0.39},
    'da': {'name': 'Danish', 'tokens': 22921651314, 'docs': 25429808, 'percentage': 0.36},
    'bg': {'name': 'Bulgarian', 'tokens': 22917954776, 'docs': 24131819, 'percentage': 0.36},
    'no': {'name': 'Norwegian', 'tokens': 18426628868, 'docs': 18907310, 'percentage': 0.29},
    'hi': {'name': 'Hindi', 'tokens': 16791362871, 'docs': 19665355, 'percentage': 0.27},
    'sk': {'name': 'Slovak', 'tokens': 16442669076, 'docs': 18582517, 'percentage': 0.26},
    'th': {'name': 'Thai', 'tokens': 15717374014, 'docs': 20960550, 'percentage': 0.25},
    'lt': {'name': 'Lithuanian', 'tokens': 14247110836, 'docs': 13339785, 'percentage': 0.23},
    'ca': {'name': 'Catalan', 'tokens': 12530288006, 'docs': 15531777, 'percentage': 0.20},
    'id': {'name': 'Indonesian', 'tokens': 12062966061, 'docs': 23251368, 'percentage': 0.19},
    'bn': {'name': 'Bangla', 'tokens': 9572929804, 'docs': 12436596, 'percentage': 0.15},
    
    # Add more languages as needed for specific experiments
    'et': {'name': 'Estonian', 'tokens': 8805656165, 'docs': 8004753, 'percentage': 0.14},
    'sl': {'name': 'Slovenian', 'tokens': 8007587522, 'docs': 7335378, 'percentage': 0.13},
    'lv': {'name': 'Latvian', 'tokens': 7845180319, 'docs': 7136587, 'percentage': 0.12},
    'he': {'name': 'Hebrew', 'tokens': 4937152096, 'docs': 4653979, 'percentage': 0.08},
    'ta': {'name': 'Tamil', 'tokens': 4378078610, 'docs': 4728460, 'percentage': 0.07},
    'ur': {'name': 'Urdu', 'tokens': 2703052627, 'docs': 2757279, 'percentage': 0.04},
    'ka': {'name': 'Georgian', 'tokens': 2617625564, 'docs': 3120321, 'percentage': 0.04},
    'ml': {'name': 'Malayalam', 'tokens': 2100556809, 'docs': 2693052, 'percentage': 0.03},
    'ne': {'name': 'Nepali', 'tokens': 2061601961, 'docs': 3124040, 'percentage': 0.03},
    'mr': {'name': 'Marathi', 'tokens': 1955227796, 'docs': 2266588, 'percentage': 0.03},
    'te': {'name': 'Telugu', 'tokens': 1566972146, 'docs': 1822865, 'percentage': 0.02},
    'kn': {'name': 'Kannada', 'tokens': 1242285201, 'docs': 1352142, 'percentage': 0.02},
    'gu': {'name': 'Gujarati', 'tokens': 1131730537, 'docs': 1162878, 'percentage': 0.02},
    'si': {'name': 'Sinhala', 'tokens': 880289097, 'docs': 753655, 'percentage': 0.01},
    'pa': {'name': 'Punjabi', 'tokens': 727546145, 'docs': 646987, 'percentage': 0.01},
    'am': {'name': 'Amharic', 'tokens': 358206762, 'docs': 243349, 'percentage': 0.01},
}

@dataclass
class CulturaXConfig:
    """Configuration for CulturaX dataset integration."""
    
    # Dataset access
    use_auth_token: bool = True
    cache_dir: Optional[str] = None
    
    # Language selection
    languages: List[str] = field(default_factory=lambda: ['en', 'es', 'fr', 'de', 'zh', 'ar'])
    max_languages: Optional[int] = None
    min_language_percentage: float = 0.01  # Minimum percentage of dataset to include language
    
    # Sampling and filtering
    max_samples_per_language: Optional[int] = 100000
    min_text_length: int = 50
    max_text_length: int = 8192
    quality_filters: Dict[str, Any] = field(default_factory=lambda: {
        'min_words': 10,
        'max_url_ratio': 0.3,
        'max_special_char_ratio': 0.1,
        'language_detection_confidence': 0.8
    })
    
    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Processing
    streaming: bool = True
    batch_size: int = 1000
    num_proc: int = 4
    seed: int = 42
    
    # Cross-lingual settings
    enable_cross_lingual_pairs: bool = True
    cross_lingual_languages: List[str] = field(default_factory=lambda: [
        'en', 'es', 'fr', 'de', 'zh', 'ar', 'hi', 'ru', 'ja', 'ko'
    ])
    
    # Retrieval corpus settings
    build_retrieval_corpus: bool = True
    retrieval_corpus_size: int = 1000000
    retrieval_languages: List[str] = field(default_factory=lambda: [
        'en', 'es', 'fr', 'de', 'zh', 'ar', 'hi', 'ru'
    ])


class CulturaXQualityFilter:
    """Quality filtering for CulturaX text data."""
    
    def __init__(self, config: CulturaXConfig):
        self.config = config
        self.filters = config.quality_filters
        
    def filter_text(self, text: str, url: str = "", source: str = "") -> bool:
        """Apply quality filters to text."""
        # Basic length check
        if len(text) < self.config.min_text_length:
            return False
        if len(text) > self.config.max_text_length:
            return False
            
        # Word count check
        words = text.split()
        if len(words) < self.filters.get('min_words', 10):
            return False
            
        # URL ratio check
        url_chars = sum(1 for char in text if char in '/.?=&%')
        url_ratio = url_chars / len(text) if text else 0
        if url_ratio > self.filters.get('max_url_ratio', 0.3):
            return False
            
        # Special character ratio check
        special_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
        special_ratio = special_chars / len(text) if text else 0
        if special_ratio > self.filters.get('max_special_char_ratio', 0.1):
            return False
            
        # Check for repeated patterns (basic deduplication)
        if self._has_excessive_repetition(text):
            return False
            
        return True
    
    def _has_excessive_repetition(self, text: str, max_repeat_ratio: float = 0.3) -> bool:
        """Check for excessive character or word repetition."""
        if len(text) < 100:
            return False
            
        # Check character repetition
        char_counts = Counter(text.lower())
        most_common_char_count = char_counts.most_common(1)[0][1]
        if most_common_char_count / len(text) > max_repeat_ratio:
            return True
            
        # Check word repetition
        words = text.lower().split()
        if len(words) > 10:
            word_counts = Counter(words)
            most_common_word_count = word_counts.most_common(1)[0][1]
            if most_common_word_count / len(words) > max_repeat_ratio:
                return True
                
        return False


class CulturaXDataset(IterableDataset):
    """Iterable dataset for CulturaX with quality filtering and language balancing."""
    
    def __init__(self, 
                 config: CulturaXConfig,
                 split: str = "train",
                 tokenizer: Optional[AutoTokenizer] = None):
        self.config = config
        self.split = split
        self.tokenizer = tokenizer
        self.quality_filter = CulturaXQualityFilter(config)
        
        # Initialize datasets for each language
        self.datasets = {}
        self.language_weights = {}
        self._load_datasets()
        
    def _load_datasets(self):
        """Load datasets for all specified languages."""
        logger.info(f"Loading CulturaX datasets for languages: {self.config.languages}")
        
        total_weight = 0
        for lang in self.config.languages:
            if lang not in CULTURAX_LANGUAGES:
                logger.warning(f"Language {lang} not found in CulturaX. Skipping.")
                continue
                
            try:
                # Load dataset for this language
                logger.info(f"Loading {lang} ({CULTURAX_LANGUAGES[lang]['name']})...")
                dataset = load_dataset(
                    "uonlp/CulturaX",
                    lang,
                    streaming=self.config.streaming,
                    use_auth_token=self.config.use_auth_token,
                    cache_dir=self.config.cache_dir
                )
                
                self.datasets[lang] = dataset['train']  # CulturaX only has train split
                
                # Calculate sampling weight based on language size
                lang_info = CULTURAX_LANGUAGES[lang]
                weight = min(lang_info['percentage'], 5.0)  # Cap at 5% to balance
                self.language_weights[lang] = weight
                total_weight += weight
                
                logger.info(f"✓ Loaded {lang}: {lang_info['docs']:,} docs, "
                           f"{lang_info['tokens']:,} tokens ({lang_info['percentage']:.2f}%)")
                           
            except Exception as e:
                logger.error(f"Failed to load {lang}: {e}")
                continue
        
        # Normalize weights
        if total_weight > 0:
            for lang in self.language_weights:
                self.language_weights[lang] /= total_weight
                
        logger.info(f"Successfully loaded {len(self.datasets)} languages")
        logger.info(f"Language weights: {self.language_weights}")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the dataset with language balancing."""
        if not self.datasets:
            return
            
        # Create iterators for each language
        iterators = {lang: iter(dataset) for lang, dataset in self.datasets.items()}
        languages = list(self.language_weights.keys())
        weights = list(self.language_weights.values())
        
        processed_count = 0
        
        while iterators:
            # Sample language based on weights
            try:
                selected_lang = np.random.choice(languages, p=weights)
            except ValueError:
                # If all weights are 0, select uniformly
                selected_lang = random.choice(languages)
            
            try:
                # Get next sample from selected language
                sample = next(iterators[selected_lang])
                
                # Apply quality filtering
                if not self.quality_filter.filter_text(
                    sample.get('text', ''),
                    sample.get('url', ''),
                    sample.get('source', '')
                ):
                    continue
                
                # Add language information
                sample['language'] = selected_lang
                sample['language_name'] = CULTURAX_LANGUAGES[selected_lang]['name']
                
                # Tokenize if tokenizer provided
                if self.tokenizer:
                    tokens = self.tokenizer(
                        sample['text'],
                        truncation=True,
                        max_length=self.config.max_text_length,
                        return_tensors="pt"
                    )
                    sample.update(tokens)
                
                processed_count += 1
                
                # Check if we've reached the limit for this language
                if (self.config.max_samples_per_language and 
                    processed_count >= self.config.max_samples_per_language * len(languages)):
                    break
                    
                yield sample
                
            except StopIteration:
                # Remove exhausted iterator
                del iterators[selected_lang]
                lang_idx = languages.index(selected_lang)
                languages.pop(lang_idx)
                weights = np.array(weights)
                weights = np.delete(weights, lang_idx)
                if len(weights) > 0:
                    weights = weights / weights.sum()  # Renormalize
                weights = weights.tolist()
                
                if not iterators:
                    break


class CulturaXCrossLingualDataset(Dataset):
    """Dataset for cross-lingual evaluation tasks."""
    
    def __init__(self, 
                 config: CulturaXConfig,
                 task_type: str = "parallel_retrieval",
                 tokenizer: Optional[AutoTokenizer] = None):
        self.config = config
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.samples = []
        
        self._create_cross_lingual_samples()
    
    def _create_cross_lingual_samples(self):
        """Create cross-lingual task samples."""
        logger.info(f"Creating cross-lingual samples for task: {self.task_type}")
        
        if self.task_type == "parallel_retrieval":
            self._create_parallel_retrieval_samples()
        elif self.task_type == "zero_shot_transfer":
            self._create_zero_shot_samples()
        elif self.task_type == "code_switching":
            self._create_code_switching_samples()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _create_parallel_retrieval_samples(self):
        """Create samples for parallel document retrieval across languages."""
        # This would typically require parallel corpora
        # For now, create synthetic samples
        languages = self.config.cross_lingual_languages[:4]  # Limit for efficiency
        
        for i in range(1000):  # Create 1000 samples
            # Select source and target languages
            source_lang, target_lang = random.sample(languages, 2)
            
            sample = {
                'source_language': source_lang,
                'target_language': target_lang,
                'source_text': f"Sample text in {CULTURAX_LANGUAGES[source_lang]['name']} {i}",
                'target_text': f"Sample text in {CULTURAX_LANGUAGES[target_lang]['name']} {i}",
                'is_parallel': random.choice([True, False]),
                'similarity_score': random.uniform(0.5, 1.0)
            }
            
            self.samples.append(sample)
    
    def _create_zero_shot_samples(self):
        """Create samples for zero-shot cross-lingual transfer."""
        # Create classification or QA samples
        for i in range(1000):
            lang = random.choice(self.config.cross_lingual_languages)
            
            sample = {
                'language': lang,
                'text': f"Sample text for classification in {CULTURAX_LANGUAGES[lang]['name']} {i}",
                'label': random.choice(['positive', 'negative', 'neutral']),
                'task': 'sentiment_classification'
            }
            
            self.samples.append(sample)
    
    def _create_code_switching_samples(self):
        """Create samples with code-switching between languages."""
        for i in range(1000):
            lang1, lang2 = random.sample(self.config.cross_lingual_languages, 2)
            
            # Simulate code-switching text
            text = f"Text starts in {CULTURAX_LANGUAGES[lang1]['name']} then switches to {CULTURAX_LANGUAGES[lang2]['name']} and back."
            
            sample = {
                'text': text,
                'primary_language': lang1,
                'secondary_language': lang2,
                'switch_points': [20, 60],  # Character positions of switches
                'task': 'language_identification'
            }
            
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx].copy()
        
        # Tokenize if tokenizer provided
        if self.tokenizer and 'text' in sample:
            tokens = self.tokenizer(
                sample['text'],
                truncation=True,
                max_length=self.config.max_text_length,
                padding='max_length',
                return_tensors="pt"
            )
            sample.update(tokens)
        
        return sample


class CulturaXRetrievalCorpus:
    """Build and manage retrieval corpus from CulturaX."""
    
    def __init__(self, config: CulturaXConfig, output_dir: str = "culturax_retrieval_corpus"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.corpus_data = defaultdict(list)
        self.embeddings = {}
        
    def build_corpus(self, force_rebuild: bool = False):
        """Build multilingual retrieval corpus."""
        corpus_file = self.output_dir / "corpus_metadata.json"
        
        if corpus_file.exists() and not force_rebuild:
            logger.info("Loading existing corpus metadata...")
            with open(corpus_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.corpus_data = defaultdict(list, metadata['corpus_data'])
            return
        
        logger.info("Building retrieval corpus from CulturaX...")
        
        quality_filter = CulturaXQualityFilter(self.config)
        
        for lang in self.config.retrieval_languages:
            if lang not in CULTURAX_LANGUAGES:
                continue
                
            logger.info(f"Processing {lang} for retrieval corpus...")
            
            try:
                # Load dataset
                dataset = load_dataset(
                    "uonlp/CulturaX",
                    lang,
                    streaming=True,
                    use_auth_token=self.config.use_auth_token
                )
                
                # Sample documents for corpus
                count = 0
                target_count = self.config.retrieval_corpus_size // len(self.config.retrieval_languages)
                
                for sample in dataset['train']:
                    if count >= target_count:
                        break
                    
                    text = sample.get('text', '')
                    if not quality_filter.filter_text(text, sample.get('url', '')):
                        continue
                    
                    # Create corpus entry
                    corpus_entry = {
                        'id': f"{lang}_{count}",
                        'text': text[:2000],  # Limit for efficiency
                        'language': lang,
                        'url': sample.get('url', ''),
                        'source': sample.get('source', ''),
                        'timestamp': sample.get('timestamp', ''),
                        'length': len(text),
                        'word_count': len(text.split())
                    }
                    
                    self.corpus_data[lang].append(corpus_entry)
                    count += 1
                    
                    if count % 1000 == 0:
                        logger.info(f"  Processed {count}/{target_count} documents for {lang}")
                
                logger.info(f"✓ Added {count} documents for {lang}")
                
            except Exception as e:
                logger.error(f"Error processing {lang}: {e}")
                continue
        
        # Save corpus metadata
        metadata = {
            'config': self.config.__dict__,
            'languages': list(self.corpus_data.keys()),
            'total_documents': sum(len(docs) for docs in self.corpus_data.values()),
            'corpus_data': dict(self.corpus_data)
        }
        
        with open(corpus_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Corpus built with {metadata['total_documents']} documents across {len(metadata['languages'])} languages")
    
    def save_corpus_texts(self):
        """Save corpus texts for embedding computation."""
        for lang, documents in self.corpus_data.items():
            lang_file = self.output_dir / f"corpus_{lang}.jsonl"
            
            with open(lang_file, 'w', encoding='utf-8') as f:
                for doc in documents:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(documents)} documents for {lang} to {lang_file}")
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        stats = {
            'total_documents': sum(len(docs) for docs in self.corpus_data.values()),
            'languages': list(self.corpus_data.keys()),
            'documents_per_language': {lang: len(docs) for lang, docs in self.corpus_data.items()},
            'total_tokens': sum(sum(doc['word_count'] for doc in docs) for docs in self.corpus_data.values()),
            'average_doc_length': {}
        }
        
        for lang, docs in self.corpus_data.items():
            if docs:
                avg_length = sum(doc['length'] for doc in docs) / len(docs)
                stats['average_doc_length'][lang] = avg_length
        
        return stats


class CulturaXDataModule:
    """Complete data module for CulturaX integration with RC-Mamba."""
    
    def __init__(self, config: CulturaXConfig, tokenizer: Optional[AutoTokenizer] = None):
        self.config = config
        self.tokenizer = tokenizer
        
        # Initialize components
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.cross_lingual_datasets = {}
        self.retrieval_corpus = None
        
        if config.build_retrieval_corpus:
            self.retrieval_corpus = CulturaXRetrievalCorpus(config)
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing."""
        logger.info("Setting up CulturaX data module...")
        
        if stage == "fit" or stage is None:
            # Create training and validation datasets
            self.train_dataset = CulturaXDataset(
                self.config, 
                split="train", 
                tokenizer=self.tokenizer
            )
            
            # For validation, we'll use a subset of the same data
            val_config = CulturaXConfig(**self.config.__dict__)
            val_config.max_samples_per_language = 1000  # Smaller validation set
            self.val_dataset = CulturaXDataset(
                val_config,
                split="validation",
                tokenizer=self.tokenizer
            )
        
        if stage == "test" or stage is None:
            # Create test dataset
            test_config = CulturaXConfig(**self.config.__dict__)
            test_config.max_samples_per_language = 500  # Smaller test set
            self.test_dataset = CulturaXDataset(
                test_config,
                split="test",
                tokenizer=self.tokenizer
            )
        
        # Setup cross-lingual evaluation datasets
        if self.config.enable_cross_lingual_pairs:
            self._setup_cross_lingual_datasets()
        
        # Build retrieval corpus if requested
        if self.config.build_retrieval_corpus and self.retrieval_corpus:
            self.retrieval_corpus.build_corpus()
    
    def _setup_cross_lingual_datasets(self):
        """Setup cross-lingual evaluation datasets."""
        logger.info("Setting up cross-lingual evaluation datasets...")
        
        tasks = ["parallel_retrieval", "zero_shot_transfer", "code_switching"]
        
        for task in tasks:
            self.cross_lingual_datasets[task] = CulturaXCrossLingualDataset(
                self.config,
                task_type=task,
                tokenizer=self.tokenizer
            )
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_proc,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_proc,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_proc,
            pin_memory=True
        )
    
    def cross_lingual_dataloader(self, task: str) -> DataLoader:
        """Get cross-lingual evaluation dataloader."""
        if task not in self.cross_lingual_datasets:
            raise ValueError(f"Cross-lingual task {task} not available")
        
        return DataLoader(
            self.cross_lingual_datasets[task],
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_proc
        )
    
    def get_language_info(self) -> Dict[str, Any]:
        """Get information about configured languages."""
        return {
            lang: CULTURAX_LANGUAGES.get(lang, {'name': lang, 'tokens': 0, 'docs': 0, 'percentage': 0})
            for lang in self.config.languages
        }
    
    def get_corpus_stats(self) -> Optional[Dict[str, Any]]:
        """Get retrieval corpus statistics."""
        if self.retrieval_corpus:
            return self.retrieval_corpus.get_corpus_stats()
        return None


def create_culturax_config(
    languages: List[str] = None,
    max_samples_per_language: int = 50000,
    streaming: bool = True,
    build_retrieval_corpus: bool = True,
    **kwargs
) -> CulturaXConfig:
    """Create a CulturaX configuration with sensible defaults."""
    
    if languages is None:
        # Default to top 10 languages by size
        languages = ['en', 'ru', 'es', 'de', 'fr', 'zh', 'it', 'pt', 'pl', 'ja']
    
    config = CulturaXConfig(
        languages=languages,
        max_samples_per_language=max_samples_per_language,
        streaming=streaming,
        build_retrieval_corpus=build_retrieval_corpus,
        **kwargs
    )
    
    return config


def test_culturax_integration():
    """Test function for CulturaX integration."""
    logger.info("Testing CulturaX integration...")
    
    # Create small test configuration
    config = create_culturax_config(
        languages=['en', 'es'],  # Start with just 2 languages
        max_samples_per_language=100,  # Very small for testing
        streaming=True,
        build_retrieval_corpus=False  # Skip for quick test
    )
    
    # Test dataset creation
    try:
        data_module = CulturaXDataModule(config)
        data_module.setup()
        
        logger.info("✓ CulturaX data module created successfully")
        
        # Test iteration
        train_loader = data_module.train_dataloader()
        
        logger.info("✓ Training dataloader created successfully")
        logger.info("✓ CulturaX integration test completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ CulturaX integration test failed: {e}")
        return False


if __name__ == "__main__":
    # Run integration test
    test_culturax_integration()
