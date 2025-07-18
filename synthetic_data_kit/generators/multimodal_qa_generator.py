# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Multimodal Question Answering Generator

import os
from typing import Optional

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import load_config, get_generation_config

class MultimodalQAGenerator:
    """Generates Multimodal Question Answering data (text QA from text+image context)"""
    def __init__(self, client: LLMClient, config_path: Optional[str] = None):
        self.client = client
        self.config = load_config(str(config_path) if config_path else None) if config_path else client.config
        self.generation_config = get_generation_config(self.config)

    def process_dataset(self, documents, output_dir: str, num_examples=None, verbose=False) -> str:
        # documents: list of dicts with 'question' and 'answer'
        qa_pairs = []
        for example in documents:
            qa_pairs.append({
                "question": example.get("question"),
                "answer": example.get("answer")
            })
        output_path = os.path.join(output_dir, "multimodal_qa_pairs.json")
        with open(output_path, "w", encoding="utf-8") as f:
            import json
            json.dump({"qa_pairs": qa_pairs}, f, indent=2)
        if verbose:
            print(f"Saved processed multimodal QA pairs to {output_path}")
        return output_path 