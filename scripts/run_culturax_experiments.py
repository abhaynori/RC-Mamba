"""
CulturaX Experiment Runner for RC-Mamba.

This script orchestrates comprehensive experiments using the CulturaX dataset,
including:

- Systematic training across different language configurations
- Ablation studies on multilingual components
- Cross-lingual transfer experiments
- Scaling experiments across model sizes
- Performance analysis and comparison
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add RC-Mamba to path
sys.path.append(str(Path(__file__).parent.parent))

from rc_mamba.data.culturax_integration import CULTURAX_LANGUAGES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CulturaXExperimentConfig:
    """Configuration for CulturaX experiments."""
    
    def __init__(self):
        # Language groups for systematic experiments
        self.language_groups = {
            'major_european': ['en', 'es', 'fr', 'de', 'it', 'pt'],
            'major_asian': ['zh', 'ja', 'ko', 'hi', 'th', 'vi'],
            'slavic': ['ru', 'pl', 'cs', 'uk', 'bg'],
            'germanic': ['en', 'de', 'nl', 'sv', 'da', 'no'],
            'romance': ['es', 'fr', 'it', 'pt', 'ro', 'ca'],
            'diverse_medium': ['ar', 'tr', 'fa', 'he', 'fi', 'hu'],
            'top_10': ['en', 'ru', 'es', 'de', 'fr', 'zh', 'it', 'pt', 'pl', 'ja'],
            'comprehensive': [
                'en', 'ru', 'es', 'de', 'fr', 'zh', 'it', 'pt', 'pl', 'ja',
                'nl', 'ar', 'tr', 'cs', 'vi', 'fa', 'hu', 'el', 'ro', 'sv',
                'uk', 'fi', 'ko', 'da', 'bg', 'no', 'hi', 'sk', 'th', 'lt'
            ]
        }
        
        # Model size configurations
        self.model_sizes = {
            'small': {
                'd_model': 256,
                'n_layers': 6,
                'batch_size': 32,
                'max_samples_per_language': 10000
            },
            'base': {
                'd_model': 512,
                'n_layers': 8,
                'batch_size': 16,
                'max_samples_per_language': 50000
            },
            'medium': {
                'd_model': 768,
                'n_layers': 12,
                'batch_size': 12,
                'max_samples_per_language': 100000
            },
            'large': {
                'd_model': 1024,
                'n_layers': 16,
                'batch_size': 8,
                'max_samples_per_language': 200000
            }
        }
        
        # Experiment types
        self.experiment_types = {
            'language_scaling': {
                'description': 'Scale across different language sets',
                'language_groups': ['major_european', 'major_asian', 'top_10', 'comprehensive'],
                'model_size': 'base',
                'training_epochs': 2
            },
            'model_scaling': {
                'description': 'Scale across different model sizes',
                'language_group': 'top_10',
                'model_sizes': ['small', 'base', 'medium'],
                'training_epochs': 2
            },
            'ablation_study': {
                'description': 'Ablation study on RC-Mamba components',
                'language_group': 'major_european',
                'model_size': 'base',
                'components': [
                    'film_conditioning',
                    'multi_hop_retrieval', 
                    'pi_dpo_training',
                    'progressive_languages'
                ],
                'training_epochs': 1
            },
            'cross_lingual_transfer': {
                'description': 'Cross-lingual transfer experiments',
                'source_languages': ['en'],
                'target_language_groups': ['romance', 'germanic', 'slavic'],
                'model_size': 'base',
                'training_epochs': 1
            },
            'long_context_study': {
                'description': 'Long context performance across languages',
                'language_group': 'major_european',
                'context_lengths': [1024, 2048, 4096, 8192],
                'model_size': 'base',
                'training_epochs': 1
            }
        }
        
        # Base training configuration
        self.base_training_config = {
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'mixed_precision': True,
            'gradient_accumulation_steps': 4,
            'save_steps': 2500,
            'eval_steps': 1250,
            'logging_steps': 100
        }
        
        # Evaluation configuration
        self.evaluation_config = {
            'tasks': [
                'perplexity',
                'zero_shot_classification',
                'cross_lingual_retrieval',
                'long_context_multilingual'
            ],
            'sequence_lengths': [512, 1024, 2048, 4096],
            'max_eval_samples': 500
        }


class CulturaXExperimentRunner:
    """Orchestrates comprehensive CulturaX experiments."""
    
    def __init__(self, 
                 config: CulturaXExperimentConfig,
                 base_output_dir: str = "culturax_experiments",
                 max_parallel_jobs: int = 2):
        self.config = config
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.max_parallel_jobs = max_parallel_jobs
        
        # Results tracking
        self.experiment_results = {}
        self.experiment_metadata = {}
        
        # Paths
        self.train_script = Path(__file__).parent / "train_culturax.py"
        self.eval_script = Path(__file__).parent / "evaluate_culturax.py"
        
        logger.info(f"Initialized experiment runner with output dir: {base_output_dir}")
    
    def run_all_experiments(self, experiment_types: List[str] = None):
        """Run all configured experiments."""
        if experiment_types is None:
            experiment_types = list(self.config.experiment_types.keys())
        
        logger.info(f"Running experiments: {experiment_types}")
        
        for exp_type in experiment_types:
            if exp_type not in self.config.experiment_types:
                logger.warning(f"Unknown experiment type: {exp_type}")
                continue
            
            logger.info(f"Starting experiment: {exp_type}")
            
            try:
                if exp_type == 'language_scaling':
                    self._run_language_scaling_experiment()
                elif exp_type == 'model_scaling':
                    self._run_model_scaling_experiment()
                elif exp_type == 'ablation_study':
                    self._run_ablation_study()
                elif exp_type == 'cross_lingual_transfer':
                    self._run_cross_lingual_transfer()
                elif exp_type == 'long_context_study':
                    self._run_long_context_study()
                else:
                    logger.warning(f"Experiment {exp_type} not implemented")
                    continue
                
                logger.info(f"✓ Completed experiment: {exp_type}")
                
            except Exception as e:
                logger.error(f"✗ Failed experiment {exp_type}: {e}")
                self.experiment_results[exp_type] = {'error': str(e)}
        
        # Generate comprehensive analysis
        self._generate_comprehensive_analysis()
        
        # Save all results
        self._save_experiment_results()
        
        logger.info("All experiments completed!")
    
    def _run_language_scaling_experiment(self):
        """Run language scaling experiments."""
        exp_config = self.config.experiment_types['language_scaling']
        results = {}
        
        for lang_group_name in exp_config['language_groups']:
            lang_group = self.config.language_groups[lang_group_name]
            
            logger.info(f"  Training on language group: {lang_group_name} ({len(lang_group)} languages)")
            
            # Create experiment directory
            exp_dir = self.base_output_dir / "language_scaling" / lang_group_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Train model
            model_path = self._train_model(
                experiment_name=f"lang_scaling_{lang_group_name}",
                languages=lang_group,
                model_size=exp_config['model_size'],
                epochs=exp_config['training_epochs'],
                output_dir=exp_dir / "training"
            )
            
            if model_path:
                # Evaluate model
                eval_results = self._evaluate_model(
                    model_path=model_path,
                    experiment_name=f"lang_scaling_{lang_group_name}",
                    languages=lang_group,
                    output_dir=exp_dir / "evaluation"
                )
                
                results[lang_group_name] = {
                    'languages': lang_group,
                    'num_languages': len(lang_group),
                    'model_path': str(model_path),
                    'evaluation_results': eval_results
                }
            else:
                results[lang_group_name] = {'error': 'Training failed'}
        
        self.experiment_results['language_scaling'] = results
    
    def _run_model_scaling_experiment(self):
        """Run model scaling experiments."""
        exp_config = self.config.experiment_types['model_scaling']
        results = {}
        
        lang_group = self.config.language_groups[exp_config['language_group']]
        
        for model_size in exp_config['model_sizes']:
            logger.info(f"  Training {model_size} model")
            
            # Create experiment directory
            exp_dir = self.base_output_dir / "model_scaling" / model_size
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Train model
            model_path = self._train_model(
                experiment_name=f"model_scaling_{model_size}",
                languages=lang_group,
                model_size=model_size,
                epochs=exp_config['training_epochs'],
                output_dir=exp_dir / "training"
            )
            
            if model_path:
                # Evaluate model
                eval_results = self._evaluate_model(
                    model_path=model_path,
                    experiment_name=f"model_scaling_{model_size}",
                    languages=lang_group,
                    output_dir=exp_dir / "evaluation"
                )
                
                # Get model statistics
                model_config = self.config.model_sizes[model_size]
                total_params = self._estimate_model_parameters(model_config)
                
                results[model_size] = {
                    'model_config': model_config,
                    'estimated_parameters': total_params,
                    'model_path': str(model_path),
                    'evaluation_results': eval_results
                }
            else:
                results[model_size] = {'error': 'Training failed'}
        
        self.experiment_results['model_scaling'] = results
    
    def _run_ablation_study(self):
        """Run ablation study on RC-Mamba components."""
        exp_config = self.config.experiment_types['ablation_study']
        results = {}
        
        lang_group = self.config.language_groups[exp_config['language_group']]
        
        # Full model (baseline)
        logger.info("  Training full model (baseline)")
        
        exp_dir = self.base_output_dir / "ablation_study" / "full_model"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        full_model_path = self._train_model(
            experiment_name="ablation_full",
            languages=lang_group,
            model_size=exp_config['model_size'],
            epochs=exp_config['training_epochs'],
            output_dir=exp_dir / "training",
            special_config={'ablation_baseline': True}
        )
        
        if full_model_path:
            full_eval_results = self._evaluate_model(
                model_path=full_model_path,
                experiment_name="ablation_full",
                languages=lang_group,
                output_dir=exp_dir / "evaluation"
            )
            
            results['full_model'] = {
                'description': 'Full RC-Mamba model',
                'model_path': str(full_model_path),
                'evaluation_results': full_eval_results
            }
        
        # Ablated models
        for component in exp_config['components']:
            logger.info(f"  Training model without {component}")
            
            exp_dir = self.base_output_dir / "ablation_study" / f"no_{component}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = self._train_model(
                experiment_name=f"ablation_no_{component}",
                languages=lang_group,
                model_size=exp_config['model_size'],
                epochs=exp_config['training_epochs'],
                output_dir=exp_dir / "training",
                special_config={'disable_component': component}
            )
            
            if model_path:
                eval_results = self._evaluate_model(
                    model_path=model_path,
                    experiment_name=f"ablation_no_{component}",
                    languages=lang_group,
                    output_dir=exp_dir / "evaluation"
                )
                
                results[f'no_{component}'] = {
                    'description': f'Model without {component}',
                    'disabled_component': component,
                    'model_path': str(model_path),
                    'evaluation_results': eval_results
                }
            else:
                results[f'no_{component}'] = {'error': 'Training failed'}
        
        self.experiment_results['ablation_study'] = results
    
    def _run_cross_lingual_transfer(self):
        """Run cross-lingual transfer experiments."""
        exp_config = self.config.experiment_types['cross_lingual_transfer']
        results = {}
        
        for source_lang in exp_config['source_languages']:
            source_results = {}
            
            # Train source model
            logger.info(f"  Training source model on {source_lang}")
            
            source_exp_dir = self.base_output_dir / "cross_lingual_transfer" / f"source_{source_lang}"
            source_exp_dir.mkdir(parents=True, exist_ok=True)
            
            source_model_path = self._train_model(
                experiment_name=f"transfer_source_{source_lang}",
                languages=[source_lang],
                model_size=exp_config['model_size'],
                epochs=exp_config['training_epochs'],
                output_dir=source_exp_dir / "training"
            )
            
            if not source_model_path:
                results[source_lang] = {'error': 'Source model training failed'}
                continue
            
            # Test transfer to different language groups
            for target_group_name in exp_config['target_language_groups']:
                target_group = self.config.language_groups[target_group_name]
                
                logger.info(f"    Evaluating transfer to {target_group_name}")
                
                # Evaluate on target languages
                target_eval_results = self._evaluate_model(
                    model_path=source_model_path,
                    experiment_name=f"transfer_{source_lang}_to_{target_group_name}",
                    languages=target_group,
                    output_dir=source_exp_dir / f"eval_{target_group_name}"
                )
                
                source_results[target_group_name] = {
                    'target_languages': target_group,
                    'evaluation_results': target_eval_results
                }
            
            results[source_lang] = {
                'source_model_path': str(source_model_path),
                'transfer_results': source_results
            }
        
        self.experiment_results['cross_lingual_transfer'] = results
    
    def _run_long_context_study(self):
        """Run long context performance study."""
        exp_config = self.config.experiment_types['long_context_study']
        results = {}
        
        lang_group = self.config.language_groups[exp_config['language_group']]
        
        # Train model
        logger.info("  Training model for long context study")
        
        exp_dir = self.base_output_dir / "long_context_study"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = self._train_model(
            experiment_name="long_context_study",
            languages=lang_group,
            model_size=exp_config['model_size'],
            epochs=exp_config['training_epochs'],
            output_dir=exp_dir / "training",
            special_config={'max_sequence_length': max(exp_config['context_lengths'])}
        )
        
        if not model_path:
            self.experiment_results['long_context_study'] = {'error': 'Training failed'}
            return
        
        # Evaluate at different context lengths
        for context_length in exp_config['context_lengths']:
            logger.info(f"    Evaluating at context length {context_length}")
            
            eval_results = self._evaluate_model(
                model_path=model_path,
                experiment_name=f"long_context_{context_length}",
                languages=lang_group,
                output_dir=exp_dir / f"eval_{context_length}",
                special_config={'max_context_length': context_length}
            )
            
            results[f'context_{context_length}'] = {
                'context_length': context_length,
                'evaluation_results': eval_results
            }
        
        self.experiment_results['long_context_study'] = {
            'model_path': str(model_path),
            'context_results': results
        }
    
    def _train_model(self,
                    experiment_name: str,
                    languages: List[str],
                    model_size: str,
                    epochs: int,
                    output_dir: Path,
                    special_config: Dict[str, Any] = None) -> Optional[Path]:
        """Train a model with specified configuration."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model configuration
        model_config = self.config.model_sizes[model_size]
        
        # Prepare training arguments
        train_args = [
            "python", str(self.train_script),
            "--output_dir", str(output_dir),
            "--languages"] + languages + [
            "--max_samples_per_language", str(model_config['max_samples_per_language']),
            "--batch_size", str(model_config['batch_size']),
            "--num_epochs", str(epochs),
            "--learning_rate", str(self.config.base_training_config['learning_rate']),
            "--streaming",
            "--progressive_languages",
            "--build_retrieval_corpus"
        ]
        
        # Add special configuration options
        if special_config:
            if special_config.get('disable_component'):
                component = special_config['disable_component']
                if component == 'film_conditioning':
                    train_args.extend(["--disable_film"])
                elif component == 'multi_hop_retrieval':
                    train_args.extend(["--disable_multihop"])
                elif component == 'pi_dpo_training':
                    train_args.extend(["--disable_pidpo"])
                elif component == 'progressive_languages':
                    train_args.remove("--progressive_languages")
        
        # Log training command
        logger.info(f"    Training command: {' '.join(train_args)}")
        
        # Run training
        try:
            result = subprocess.run(
                train_args,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode == 0:
                # Find best model path
                best_model_path = output_dir / "best_model.pt"
                if best_model_path.exists():
                    logger.info(f"    ✓ Training completed: {experiment_name}")
                    return best_model_path
                else:
                    logger.error(f"    ✗ Best model not found: {experiment_name}")
                    return None
            else:
                logger.error(f"    ✗ Training failed: {experiment_name}")
                logger.error(f"    Error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"    ✗ Training timeout: {experiment_name}")
            return None
        except Exception as e:
            logger.error(f"    ✗ Training error: {experiment_name}: {e}")
            return None
    
    def _evaluate_model(self,
                       model_path: Path,
                       experiment_name: str,
                       languages: List[str],
                       output_dir: Path,
                       special_config: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Evaluate a trained model."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare evaluation arguments
        eval_args = [
            "python", str(self.eval_script),
            str(model_path),
            "--output_dir", str(output_dir),
            "--languages"] + languages + [
            "--max_samples", str(self.config.evaluation_config['max_eval_samples']),
            "--batch_size", "8"
        ]
        
        # Add tasks
        eval_args.extend(["--tasks"] + self.config.evaluation_config['tasks'])
        
        # Log evaluation command
        logger.info(f"    Evaluation command: {' '.join(eval_args)}")
        
        # Run evaluation
        try:
            result = subprocess.run(
                eval_args,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                # Load evaluation results
                results_file = output_dir / "detailed_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        eval_results = json.load(f)
                    
                    logger.info(f"    ✓ Evaluation completed: {experiment_name}")
                    return eval_results
                else:
                    logger.error(f"    ✗ Results file not found: {experiment_name}")
                    return None
            else:
                logger.error(f"    ✗ Evaluation failed: {experiment_name}")
                logger.error(f"    Error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"    ✗ Evaluation timeout: {experiment_name}")
            return None
        except Exception as e:
            logger.error(f"    ✗ Evaluation error: {experiment_name}: {e}")
            return None
    
    def _estimate_model_parameters(self, model_config: Dict[str, Any]) -> int:
        """Estimate number of model parameters."""
        d_model = model_config['d_model']
        n_layers = model_config['n_layers']
        vocab_size = 50000  # Approximate
        
        # Rough estimation
        embedding_params = vocab_size * d_model
        layer_params = n_layers * (4 * d_model * d_model + 2 * d_model)  # Simplified
        total_params = embedding_params + layer_params
        
        return total_params
    
    def _generate_comprehensive_analysis(self):
        """Generate comprehensive analysis across all experiments."""
        logger.info("Generating comprehensive analysis...")
        
        analysis = {
            'experiment_summary': {},
            'language_analysis': {},
            'model_scaling_analysis': {},
            'component_importance': {},
            'recommendations': []
        }
        
        # Experiment summary
        total_experiments = len(self.experiment_results)
        successful_experiments = len([exp for exp in self.experiment_results.values() 
                                    if 'error' not in exp])
        
        analysis['experiment_summary'] = {
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'success_rate': successful_experiments / total_experiments if total_experiments > 0 else 0,
            'experiment_types': list(self.experiment_results.keys())
        }
        
        # Language performance analysis
        if 'language_scaling' in self.experiment_results:
            lang_perf = {}
            for group_name, results in self.experiment_results['language_scaling'].items():
                if 'evaluation_results' in results and 'perplexity' in results['evaluation_results']:
                    # Calculate average perplexity across languages
                    perplexities = []
                    for lang_results in results['evaluation_results']['perplexity'].values():
                        if 'perplexity_1024' in lang_results:
                            perplexities.append(lang_results['perplexity_1024'])
                    
                    if perplexities:
                        lang_perf[group_name] = {
                            'avg_perplexity': np.mean(perplexities),
                            'num_languages': results['num_languages'],
                            'languages': results['languages']
                        }
            
            analysis['language_analysis'] = lang_perf
        
        # Model scaling analysis
        if 'model_scaling' in self.experiment_results:
            scaling_perf = {}
            for model_size, results in self.experiment_results['model_scaling'].items():
                if 'evaluation_results' in results and 'estimated_parameters' in results:
                    # Extract performance metrics
                    eval_results = results['evaluation_results']
                    if 'perplexity' in eval_results:
                        perplexities = []
                        for lang_results in eval_results['perplexity'].values():
                            if 'perplexity_1024' in lang_results:
                                perplexities.append(lang_results['perplexity_1024'])
                        
                        if perplexities:
                            scaling_perf[model_size] = {
                                'avg_perplexity': np.mean(perplexities),
                                'parameters': results['estimated_parameters'],
                                'model_config': results['model_config']
                            }
            
            analysis['model_scaling_analysis'] = scaling_perf
        
        # Component importance analysis (from ablation study)
        if 'ablation_study' in self.experiment_results:
            ablation_results = self.experiment_results['ablation_study']
            
            if 'full_model' in ablation_results:
                baseline_perf = self._extract_performance_metric(
                    ablation_results['full_model'].get('evaluation_results')
                )
                
                component_importance = {}
                for exp_name, results in ablation_results.items():
                    if exp_name.startswith('no_') and 'evaluation_results' in results:
                        component = results.get('disabled_component', exp_name[3:])
                        ablated_perf = self._extract_performance_metric(results['evaluation_results'])
                        
                        if baseline_perf and ablated_perf:
                            # Performance degradation when component is removed
                            degradation = ablated_perf - baseline_perf
                            component_importance[component] = {
                                'performance_degradation': degradation,
                                'baseline_performance': baseline_perf,
                                'ablated_performance': ablated_perf
                            }
                
                analysis['component_importance'] = component_importance
        
        # Generate recommendations
        recommendations = []
        
        # Language recommendations
        if analysis['language_analysis']:
            best_lang_group = min(analysis['language_analysis'].items(), 
                                key=lambda x: x[1]['avg_perplexity'])
            recommendations.append(
                f"Best performing language group: {best_lang_group[0]} "
                f"(avg perplexity: {best_lang_group[1]['avg_perplexity']:.2f})"
            )
        
        # Model scaling recommendations
        if analysis['model_scaling_analysis']:
            scaling_data = [(size, data['avg_perplexity'], data['parameters']) 
                           for size, data in analysis['model_scaling_analysis'].items()]
            scaling_data.sort(key=lambda x: x[1])  # Sort by perplexity
            
            if len(scaling_data) > 1:
                best_size = scaling_data[0][0]
                recommendations.append(f"Best model size for performance: {best_size}")
        
        # Component recommendations
        if analysis['component_importance']:
            most_important = max(analysis['component_importance'].items(),
                               key=lambda x: x[1]['performance_degradation'])
            recommendations.append(
                f"Most important component: {most_important[0]} "
                f"(degradation when removed: {most_important[1]['performance_degradation']:.2f})"
            )
        
        if not recommendations:
            recommendations.append("Insufficient data for specific recommendations")
        
        analysis['recommendations'] = recommendations
        
        # Create visualizations
        self._create_experiment_visualizations(analysis)
        
        self.experiment_results['comprehensive_analysis'] = analysis
    
    def _extract_performance_metric(self, eval_results: Optional[Dict[str, Any]]) -> Optional[float]:
        """Extract a single performance metric from evaluation results."""
        if not eval_results or 'perplexity' not in eval_results:
            return None
        
        perplexities = []
        for lang_results in eval_results['perplexity'].values():
            if 'perplexity_1024' in lang_results:
                perplexities.append(lang_results['perplexity_1024'])
        
        return np.mean(perplexities) if perplexities else None
    
    def _create_experiment_visualizations(self, analysis: Dict[str, Any]):
        """Create visualizations for experiment results."""
        viz_dir = self.base_output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Language scaling plot
        if analysis['language_analysis']:
            self._plot_language_scaling(analysis['language_analysis'], viz_dir)
        
        # Model scaling plot
        if analysis['model_scaling_analysis']:
            self._plot_model_scaling(analysis['model_scaling_analysis'], viz_dir)
        
        # Component importance plot
        if analysis['component_importance']:
            self._plot_component_importance(analysis['component_importance'], viz_dir)
        
        logger.info(f"Visualizations saved to {viz_dir}")
    
    def _plot_language_scaling(self, lang_analysis: Dict[str, Any], viz_dir: Path):
        """Plot language scaling results."""
        plt.figure(figsize=(12, 8))
        
        groups = list(lang_analysis.keys())
        perplexities = [data['avg_perplexity'] for data in lang_analysis.values()]
        num_languages = [data['num_languages'] for data in lang_analysis.values()]
        
        # Create scatter plot
        plt.scatter(num_languages, perplexities, s=100, alpha=0.7)
        
        # Add labels
        for i, group in enumerate(groups):
            plt.annotate(group, (num_languages[i], perplexities[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Number of Languages')
        plt.ylabel('Average Perplexity')
        plt.title('Language Scaling: Performance vs Number of Languages')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "language_scaling.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_scaling(self, scaling_analysis: Dict[str, Any], viz_dir: Path):
        """Plot model scaling results."""
        plt.figure(figsize=(12, 8))
        
        model_sizes = list(scaling_analysis.keys())
        perplexities = [data['avg_perplexity'] for data in scaling_analysis.values()]
        parameters = [data['parameters'] / 1e6 for data in scaling_analysis.values()]  # Convert to millions
        
        # Create scatter plot
        plt.scatter(parameters, perplexities, s=100, alpha=0.7)
        
        # Add labels
        for i, size in enumerate(model_sizes):
            plt.annotate(size, (parameters[i], perplexities[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Parameters (Millions)')
        plt.ylabel('Average Perplexity')
        plt.title('Model Scaling: Performance vs Parameters')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(parameters) > 1:
            z = np.polyfit(parameters, perplexities, 1)
            p = np.poly1d(z)
            plt.plot(parameters, p(parameters), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "model_scaling.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_component_importance(self, component_analysis: Dict[str, Any], viz_dir: Path):
        """Plot component importance from ablation study."""
        plt.figure(figsize=(12, 8))
        
        components = list(component_analysis.keys())
        degradations = [data['performance_degradation'] for data in component_analysis.values()]
        
        # Create bar plot
        bars = plt.bar(components, degradations, alpha=0.7)
        
        # Color bars by importance
        colors = plt.cm.Reds([d / max(degradations) for d in degradations])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Components')
        plt.ylabel('Performance Degradation (Higher = More Important)')
        plt.title('Component Importance: Performance Loss When Removed')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "component_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_experiment_results(self):
        """Save all experiment results and metadata."""
        # Save detailed results as JSON
        results_file = self.base_output_dir / "experiment_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save experiment metadata
        metadata = {
            'experiment_date': datetime.now().isoformat(),
            'config': {
                'language_groups': self.config.language_groups,
                'model_sizes': self.config.model_sizes,
                'experiment_types': self.config.experiment_types
            },
            'total_experiments': len(self.experiment_results),
            'successful_experiments': len([exp for exp in self.experiment_results.values() 
                                         if 'error' not in exp])
        }
        
        metadata_file = self.base_output_dir / "experiment_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        self._save_summary_report()
        
        logger.info(f"Experiment results saved to {self.base_output_dir}")
    
    def _save_summary_report(self):
        """Save a human-readable summary report."""
        report_file = self.base_output_dir / "experiment_summary.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# CulturaX RC-Mamba Experiment Summary\n\n")
            f.write(f"**Experiment Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall summary
            if 'comprehensive_analysis' in self.experiment_results:
                analysis = self.experiment_results['comprehensive_analysis']
                
                f.write("## Overall Summary\n\n")
                summary = analysis.get('experiment_summary', {})
                f.write(f"- **Total Experiments:** {summary.get('total_experiments', 0)}\n")
                f.write(f"- **Successful Experiments:** {summary.get('successful_experiments', 0)}\n")
                f.write(f"- **Success Rate:** {summary.get('success_rate', 0):.1%}\n")
                f.write(f"- **Experiment Types:** {', '.join(summary.get('experiment_types', []))}\n\n")
                
                # Recommendations
                f.write("## Key Findings and Recommendations\n\n")
                for rec in analysis.get('recommendations', []):
                    f.write(f"- {rec}\n")
                f.write("\n")
                
                # Language analysis
                if analysis.get('language_analysis'):
                    f.write("## Language Performance Analysis\n\n")
                    f.write("| Language Group | Avg Perplexity | Num Languages |\n")
                    f.write("|----------------|----------------|---------------|\n")
                    
                    for group, data in analysis['language_analysis'].items():
                        f.write(f"| {group} | {data['avg_perplexity']:.2f} | {data['num_languages']} |\n")
                    f.write("\n")
                
                # Model scaling analysis
                if analysis.get('model_scaling_analysis'):
                    f.write("## Model Scaling Analysis\n\n")
                    f.write("| Model Size | Avg Perplexity | Parameters (M) |\n")
                    f.write("|------------|----------------|----------------|\n")
                    
                    for size, data in analysis['model_scaling_analysis'].items():
                        params_m = data['parameters'] / 1e6
                        f.write(f"| {size} | {data['avg_perplexity']:.2f} | {params_m:.1f} |\n")
                    f.write("\n")
                
                # Component importance
                if analysis.get('component_importance'):
                    f.write("## Component Importance (Ablation Study)\n\n")
                    f.write("| Component | Performance Degradation | Importance |\n")
                    f.write("|-----------|-------------------------|------------|\n")
                    
                    # Sort by importance
                    sorted_components = sorted(
                        analysis['component_importance'].items(),
                        key=lambda x: x[1]['performance_degradation'],
                        reverse=True
                    )
                    
                    for component, data in sorted_components:
                        degradation = data['performance_degradation']
                        importance = "High" if degradation > 5 else "Medium" if degradation > 2 else "Low"
                        f.write(f"| {component} | {degradation:.2f} | {importance} |\n")
                    f.write("\n")
            
            # Individual experiment details
            f.write("## Detailed Experiment Results\n\n")
            for exp_type, results in self.experiment_results.items():
                if exp_type == 'comprehensive_analysis':
                    continue
                
                f.write(f"### {exp_type.replace('_', ' ').title()}\n\n")
                
                if 'error' in results:
                    f.write(f"**Status:** Failed - {results['error']}\n\n")
                else:
                    f.write("**Status:** Completed successfully\n\n")
                    # Add more specific details based on experiment type
                    if exp_type == 'language_scaling':
                        f.write(f"**Language Groups Tested:** {len(results)}\n")
                    elif exp_type == 'model_scaling':
                        f.write(f"**Model Sizes Tested:** {list(results.keys())}\n")
                    elif exp_type == 'ablation_study':
                        f.write(f"**Components Tested:** {len([k for k in results.keys() if k.startswith('no_')])}\n")
                
                f.write("\n")


def main():
    """Main experiment runner function."""
    parser = argparse.ArgumentParser(description="Run comprehensive CulturaX experiments")
    
    parser.add_argument("--output_dir", type=str, default="culturax_experiments",
                       help="Base output directory for all experiments")
    parser.add_argument("--experiments", nargs="+", default=None,
                       help="Experiment types to run")
    parser.add_argument("--max_parallel_jobs", type=int, default=2,
                       help="Maximum parallel training jobs")
    parser.add_argument("--quick_test", action="store_true",
                       help="Run quick test experiments (reduced epochs and samples)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = CulturaXExperimentConfig()
    
    # Modify for quick test
    if args.quick_test:
        for exp_config in config.experiment_types.values():
            if 'training_epochs' in exp_config:
                exp_config['training_epochs'] = 1
        
        for model_config in config.model_sizes.values():
            model_config['max_samples_per_language'] = min(
                model_config['max_samples_per_language'], 1000
            )
        
        config.evaluation_config['max_eval_samples'] = 100
    
    # Create experiment runner
    runner = CulturaXExperimentRunner(
        config=config,
        base_output_dir=args.output_dir,
        max_parallel_jobs=args.max_parallel_jobs
    )
    
    # Run experiments
    runner.run_all_experiments(args.experiments)


if __name__ == "__main__":
    main()
