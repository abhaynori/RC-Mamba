"""
Comprehensive Evaluation Framework for RC-Mamba.

This module provides evaluation utilities for long-context reasoning,
multimodal understanding, cross-lingual capabilities, and efficiency analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
import json
from pathlib import Path
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metric_name: str
    score: float
    details: Dict[str, Any]
    metadata: Dict[str, Any]


class NeedleInHaystackEvaluator:
    """Needle-in-a-haystack evaluation for long-context understanding."""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_context_lengths: List[int] = [1000, 2000, 5000, 10000, 20000],
        num_samples_per_length: int = 50
    ):
        self.tokenizer = tokenizer
        self.max_context_lengths = max_context_lengths
        self.num_samples_per_length = num_samples_per_length
        
        # Needle templates
        self.needles = [
            "The secret code is {code}.",
            "Remember this important number: {code}.",
            "The password you need is {code}.",
            "Don't forget the key: {code}.",
            "The magical number is {code}."
        ]
        
        # Haystack text (common text to fill context)
        self.haystack_text = """
        In the realm of artificial intelligence and machine learning, researchers 
        have been exploring various architectures and methodologies to improve 
        model performance and efficiency. Deep learning has revolutionized many 
        fields including computer vision, natural language processing, and speech 
        recognition. The development of transformer architectures has been 
        particularly influential, leading to breakthrough models like BERT, GPT, 
        and T5. More recently, state space models have emerged as a promising 
        alternative, offering linear complexity for sequence modeling tasks.
        """
    
    def generate_sample(self, context_length: int, needle_position: float = 0.5) -> Dict[str, Any]:
        """Generate a single needle-in-haystack sample."""
        # Generate random code
        code = np.random.randint(10000, 99999)
        needle = np.random.choice(self.needles).format(code=code)
        
        # Create haystack of desired length
        target_tokens = context_length - len(self.tokenizer.encode(needle))
        haystack_tokens = self.tokenizer.encode(self.haystack_text * 100)[:target_tokens]
        haystack = self.tokenizer.decode(haystack_tokens)
        
        # Insert needle at specified position
        split_pos = int(len(haystack) * needle_position)
        context = haystack[:split_pos] + " " + needle + " " + haystack[split_pos:]
        
        # Create question
        question = f"What was the secret code mentioned in the text?"
        
        return {
            "context": context,
            "question": question,
            "answer": str(code),
            "needle_position": needle_position,
            "context_length": len(self.tokenizer.encode(context))
        }
    
    def evaluate_model(
        self, 
        model: nn.Module, 
        retriever: Optional[Any] = None
    ) -> List[EvaluationResult]:
        """Evaluate model on needle-in-haystack task."""
        results = []
        
        for context_length in self.max_context_lengths:
            correct = 0
            total = 0
            position_scores = []
            
            for _ in range(self.num_samples_per_length):
                # Test different needle positions
                for position in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    sample = self.generate_sample(context_length, position)
                    
                    # Prepare input
                    full_text = f"{sample['context']}\n\nQuestion: {sample['question']}\nAnswer:"
                    inputs = self.tokenizer.encode(full_text, return_tensors="pt")
                    
                    # Get model prediction
                    with torch.no_grad():
                        if retriever:
                            retrieval_emb = retriever({"text": sample['context']})
                            outputs = model(inputs, retrieval=retrieval_emb)
                        else:
                            outputs = model(inputs)
                        
                        # Generate answer
                        generated = model.generate(
                            inputs, 
                            max_new_tokens=10,
                            retrieval=retriever({"text": sample['context']}) if retriever else None
                        )
                        
                        answer_tokens = generated[0][len(inputs[0]):]
                        predicted_answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                    
                    # Check if correct
                    is_correct = sample['answer'] in predicted_answer
                    correct += is_correct
                    total += 1
                    position_scores.append((position, is_correct))
            
            accuracy = correct / total if total > 0 else 0
            results.append(EvaluationResult(
                metric_name=f"needle_accuracy_len_{context_length}",
                score=accuracy,
                details={"position_scores": position_scores},
                metadata={"context_length": context_length, "total_samples": total}
            ))
        
        return results


class MultimodalEvaluator:
    """Evaluator for multimodal understanding tasks."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        
    def load_vqa_dataset(self, split: str = "validation", max_samples: int = 1000):
        """Load Visual Question Answering dataset."""
        try:
            dataset = load_dataset("HuggingFaceM4/VQAv2", split=split)
            return dataset.select(range(min(len(dataset), max_samples)))
        except:
            # Return dummy data if dataset unavailable
            return self._create_dummy_vqa_data(max_samples)
    
    def _create_dummy_vqa_data(self, num_samples: int):
        """Create dummy VQA data for testing."""
        dummy_data = []
        questions = [
            "What color is the cat?",
            "How many people are in the image?",
            "What is the weather like?",
            "What objects are visible?",
            "Where is this scene taking place?"
        ]
        answers = ["orange", "two", "sunny", "car, tree, building", "park"]
        
        for i in range(num_samples):
            dummy_data.append({
                "question": questions[i % len(questions)],
                "answer": answers[i % len(answers)],
                "image": None,  # Placeholder
                "image_id": i
            })
        return dummy_data
    
    def evaluate_vqa(
        self, 
        model: nn.Module, 
        retriever: Any,
        max_samples: int = 500
    ) -> List[EvaluationResult]:
        """Evaluate on Visual Question Answering task."""
        dataset = self.load_vqa_dataset(max_samples=max_samples)
        
        correct = 0
        total = 0
        predictions = []
        ground_truths = []
        
        for sample in dataset:
            if sample["image"] is None:
                continue  # Skip samples without images
                
            # Prepare multimodal query
            query = {
                "text": sample["question"],
                "image": sample["image"]
            }
            
            # Get retrieval embedding
            retrieval_emb = retriever(query)
            
            # Prepare text input
            input_text = f"Question: {sample['question']}\nAnswer:"
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            
            # Generate answer
            with torch.no_grad():
                generated = model.generate(
                    inputs,
                    max_new_tokens=20,
                    retrieval=retrieval_emb
                )
                
                answer_tokens = generated[0][len(inputs[0]):]
                predicted_answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            predictions.append(predicted_answer)
            ground_truths.append(sample["answer"])
            
            # Simple exact match scoring
            is_correct = predicted_answer.lower() == sample["answer"].lower()
            correct += is_correct
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return [EvaluationResult(
            metric_name="vqa_accuracy",
            score=accuracy,
            details={
                "predictions": predictions[:10],  # Store first 10 for inspection
                "ground_truths": ground_truths[:10]
            },
            metadata={"total_samples": total}
        )]


class CrossLingualEvaluator:
    """Evaluator for cross-lingual understanding."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.languages = ["en", "fr", "de", "es", "zh", "ja", "ar"]
    
    def load_xnli_dataset(self, split: str = "validation", max_samples: int = 1000):
        """Load XNLI cross-lingual natural language inference dataset."""
        try:
            dataset = load_dataset("xnli", split=split)
            return dataset.select(range(min(len(dataset), max_samples)))
        except:
            return self._create_dummy_xnli_data(max_samples)
    
    def _create_dummy_xnli_data(self, num_samples: int):
        """Create dummy XNLI data for testing."""
        dummy_data = []
        premises = [
            "The cat is sleeping on the couch.",
            "It's raining outside today.",
            "The students are studying in the library.",
            "The restaurant serves delicious food.",
            "The train arrives at 5 PM."
        ]
        hypotheses = [
            "The cat is awake.",
            "The weather is nice.",
            "People are reading books.",
            "The food tastes good.",
            "The train is late."
        ]
        labels = [2, 2, 1, 1, 2]  # contradiction, contradiction, entailment, entailment, contradiction
        
        for i in range(num_samples):
            lang = self.languages[i % len(self.languages)]
            dummy_data.append({
                "premise": premises[i % len(premises)],
                "hypothesis": hypotheses[i % len(hypotheses)],
                "label": labels[i % len(labels)],
                "language": lang
            })
        return dummy_data
    
    def evaluate_xnli(
        self, 
        model: nn.Module, 
        retriever: Any,
        max_samples: int = 500
    ) -> List[EvaluationResult]:
        """Evaluate cross-lingual natural language inference."""
        dataset = self.load_xnli_dataset(max_samples=max_samples)
        
        lang_results = {lang: {"correct": 0, "total": 0} for lang in self.languages}
        overall_correct = 0
        overall_total = 0
        
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        
        for sample in dataset:
            lang = sample.get("language", "en")
            
            # Prepare input text
            input_text = f"Premise: {sample['premise']}\nHypothesis: {sample['hypothesis']}\nRelation:"
            
            # Get retrieval embedding
            query = {"text": input_text}
            retrieval_emb = retriever(query)
            
            # Tokenize input
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            
            # Generate prediction
            with torch.no_grad():
                generated = model.generate(
                    inputs,
                    max_new_tokens=10,
                    retrieval=retrieval_emb
                )
                
                answer_tokens = generated[0][len(inputs[0]):]
                predicted_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip().lower()
            
            # Map prediction to label
            predicted_label = None
            for label_id, label_text in label_map.items():
                if label_text in predicted_text:
                    predicted_label = label_id
                    break
            
            if predicted_label is not None:
                is_correct = predicted_label == sample["label"]
                lang_results[lang]["correct"] += is_correct
                overall_correct += is_correct
            
            lang_results[lang]["total"] += 1
            overall_total += 1
        
        # Calculate results
        results = []
        for lang in self.languages:
            if lang_results[lang]["total"] > 0:
                accuracy = lang_results[lang]["correct"] / lang_results[lang]["total"]
                results.append(EvaluationResult(
                    metric_name=f"xnli_accuracy_{lang}",
                    score=accuracy,
                    details={},
                    metadata={"language": lang, "total_samples": lang_results[lang]["total"]}
                ))
        
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
        results.append(EvaluationResult(
            metric_name="xnli_accuracy_overall",
            score=overall_accuracy,
            details={"language_breakdown": lang_results},
            metadata={"total_samples": overall_total}
        ))
        
        return results


class EfficiencyEvaluator:
    """Evaluator for model efficiency and computational performance."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def measure_latency(
        self, 
        model: nn.Module,
        input_lengths: List[int] = [512, 1024, 2048, 4096],
        num_runs: int = 10
    ) -> List[EvaluationResult]:
        """Measure inference latency across different input lengths."""
        model.eval()
        results = []
        
        for length in input_lengths:
            latencies = []
            
            for _ in range(num_runs):
                # Create dummy input
                dummy_input = torch.randint(0, 1000, (1, length), device=self.device)
                
                # Warm up
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # Measure latency
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                with torch.no_grad():
                    _ = model(dummy_input)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                latencies.append(end_time - start_time)
            
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            results.append(EvaluationResult(
                metric_name=f"latency_length_{length}",
                score=avg_latency,
                details={"std": std_latency, "all_latencies": latencies},
                metadata={"input_length": length, "num_runs": num_runs}
            ))
        
        return results
    
    def measure_memory_usage(
        self, 
        model: nn.Module,
        input_lengths: List[int] = [512, 1024, 2048, 4096]
    ) -> List[EvaluationResult]:
        """Measure memory usage across different input lengths."""
        if not torch.cuda.is_available():
            return []
        
        model.eval()
        results = []
        
        for length in input_lengths:
            torch.cuda.reset_peak_memory_stats()
            
            dummy_input = torch.randint(0, 1000, (1, length), device=self.device)
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            results.append(EvaluationResult(
                metric_name=f"peak_memory_length_{length}",
                score=peak_memory,
                details={},
                metadata={"input_length": length}
            ))
        
        return results


class ComprehensiveEvaluator:
    """Main evaluator that orchestrates all evaluation tasks."""
    
    def __init__(self, tokenizer: AutoTokenizer, output_dir: str = "evaluation_results"):
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize sub-evaluators
        self.needle_evaluator = NeedleInHaystackEvaluator(tokenizer)
        self.multimodal_evaluator = MultimodalEvaluator(tokenizer)
        self.crosslingual_evaluator = CrossLingualEvaluator(tokenizer)
        self.efficiency_evaluator = EfficiencyEvaluator()
    
    def run_full_evaluation(
        self, 
        model: nn.Module, 
        retriever: Any,
        save_results: bool = True
    ) -> Dict[str, List[EvaluationResult]]:
        """Run comprehensive evaluation across all tasks."""
        all_results = {}
        
        print("Running needle-in-haystack evaluation...")
        all_results["needle"] = self.needle_evaluator.evaluate_model(model, retriever)
        
        print("Running multimodal evaluation...")
        all_results["multimodal"] = self.multimodal_evaluator.evaluate_vqa(model, retriever)
        
        print("Running cross-lingual evaluation...")
        all_results["crosslingual"] = self.crosslingual_evaluator.evaluate_xnli(model, retriever)
        
        print("Running efficiency evaluation...")
        all_results["efficiency"] = (
            self.efficiency_evaluator.measure_latency(model) +
            self.efficiency_evaluator.measure_memory_usage(model)
        )
        
        if save_results:
            self.save_results(all_results)
            self.create_visualizations(all_results)
        
        return all_results
    
    def save_results(self, results: Dict[str, List[EvaluationResult]]):
        """Save evaluation results to JSON."""
        output_file = self.output_dir / "evaluation_results.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for task, task_results in results.items():
            serializable_results[task] = []
            for result in task_results:
                serializable_results[task].append({
                    "metric_name": result.metric_name,
                    "score": result.score,
                    "details": result.details,
                    "metadata": result.metadata
                })
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def create_visualizations(self, results: Dict[str, List[EvaluationResult]]):
        """Create visualization plots for evaluation results."""
        plt.style.use('seaborn')
        
        # Needle-in-haystack results
        if "needle" in results:
            self._plot_needle_results(results["needle"])
        
        # Efficiency results
        if "efficiency" in results:
            self._plot_efficiency_results(results["efficiency"])
        
        # Cross-lingual results
        if "crosslingual" in results:
            self._plot_crosslingual_results(results["crosslingual"])
    
    def _plot_needle_results(self, needle_results: List[EvaluationResult]):
        """Plot needle-in-haystack evaluation results."""
        context_lengths = []
        accuracies = []
        
        for result in needle_results:
            if "needle_accuracy_len_" in result.metric_name:
                length = result.metadata["context_length"]
                context_lengths.append(length)
                accuracies.append(result.score)
        
        plt.figure(figsize=(10, 6))
        plt.plot(context_lengths, accuracies, marker='o', linewidth=2, markersize=8)
        plt.xlabel("Context Length")
        plt.ylabel("Accuracy")
        plt.title("Needle-in-Haystack Performance vs Context Length")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / "needle_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_results(self, efficiency_results: List[EvaluationResult]):
        """Plot efficiency evaluation results."""
        latency_results = [r for r in efficiency_results if "latency" in r.metric_name]
        memory_results = [r for r in efficiency_results if "memory" in r.metric_name]
        
        if latency_results:
            lengths = [r.metadata["input_length"] for r in latency_results]
            latencies = [r.score for r in latency_results]
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(lengths, latencies, marker='o', linewidth=2, markersize=8)
            plt.xlabel("Input Length")
            plt.ylabel("Latency (seconds)")
            plt.title("Inference Latency vs Input Length")
            plt.grid(True, alpha=0.3)
        
        if memory_results:
            lengths = [r.metadata["input_length"] for r in memory_results]
            memory = [r.score for r in memory_results]
            
            plt.subplot(1, 2, 2)
            plt.plot(lengths, memory, marker='s', linewidth=2, markersize=8, color='orange')
            plt.xlabel("Input Length")
            plt.ylabel("Peak Memory (MB)")
            plt.title("Memory Usage vs Input Length")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "efficiency_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_crosslingual_results(self, crosslingual_results: List[EvaluationResult]):
        """Plot cross-lingual evaluation results."""
        lang_results = [r for r in crosslingual_results if r.metadata.get("language")]
        
        if lang_results:
            languages = [r.metadata["language"] for r in lang_results]
            accuracies = [r.score for r in lang_results]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(languages, accuracies, alpha=0.7)
            plt.xlabel("Language")
            plt.ylabel("Accuracy")
            plt.title("Cross-Lingual Performance")
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, accuracy in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{accuracy:.3f}', ha='center', va='bottom')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.savefig(self.output_dir / "crosslingual_results.png", dpi=300, bbox_inches='tight')
            plt.close()


# Example usage and testing
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Initialize tokenizer (placeholder)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(tokenizer)
    
    # Test needle evaluator
    needle_eval = NeedleInHaystackEvaluator(tokenizer, max_context_lengths=[100, 200])
    sample = needle_eval.generate_sample(200, 0.5)
    print("Sample needle-in-haystack task:")
    print(f"Context length: {len(sample['context'])}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
