"""
Computational Reproduction: Direct Preference Optimization (DPO)
Objective: Replicate the alignment of a neural model using direct preference optimization,
substantiating the mathematical formulation by Rafailov et al. (2024).
"""

import os
import torch
import logging
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

# Scientific logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DPONeurocomputationalExperiment:
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        logger.info(f"Initializing environment. Loading base model: {model_id}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Loading Tokenizer and Model in half-precision (fp16) for computational efficiency
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )

    def compile_experimental_dataset(self):
        """
        Structures the critical data (Prompt, Chosen, Rejected) extracted during the 
        research phase to simulate human feedback on biological concepts.
        """
        logger.info("Compiling experimental dataset...")
        raw_data = {
            "prompt": [
                "Describe the role of neurotransmitters.",
                "Explain cellular apoptosis."
            ],
            "chosen": [
                "Neurotransmitters are chemical messengers that transmit signals across a chemical synapse, from one neuron to the target cell.",
                "Apoptosis is a form of programmed cell death that occurs in multicellular organisms, crucial for regulating the cellular life cycle."
            ],
            "rejected": [
                "They are just brain liquids that make you think about things.",
                "It's when the cell explodes because it got too old and sick."
            ]
        }
        return Dataset.from_dict(raw_data)

    def execute_reproduction(self):
        """
        Implements the mathematical DPO loss function within the training loop.
        """
        dataset = self.compile_experimental_dataset()
        
        # Controlled reproduction parameters
        training_args = TrainingArguments(
            output_dir="./dpo_reproduction_results",
            per_device_train_batch_size=2,
            max_steps=15, # Reduced steps for Proof of Concept demonstration
            learning_rate=5e-5,
            logging_steps=1,
            fp16=True if self.device == "cuda" else False,
            remove_unused_columns=False,
            report_to="none"
        )

        logger.info("Configuring the DPO Loss Function (Beta = 0.1)")
        
        # The DPOTrainer encapsulates the target paper's mathematics
        trainer = DPOTrainer(
            self.model,
            ref_model=None, # Dynamically maintains a frozen copy of the model
            args=training_args,
            beta=0.1, 
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            max_length=256,
            max_prompt_length=128
        )

        logger.info("Initiating direct preference optimization. Monitor the DPO Loss convergence.")
        trainer.train()
        logger.info("Experiment concluded. Adjusted weights saved to the output directory.")

if __name__ == "__main__":
    experiment = DPONeurocomputationalExperiment()
    experiment.execute_reproduction()
