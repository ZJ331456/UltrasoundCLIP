#!/usr/bin/env python3
"""
Medical Term Extraction Script
Extract medical terms from refined_caption fields in JSON files using LLM
"""

import asyncio
import os
import json
import glob
from tqdm import tqdm
from openai import AsyncOpenAI
from typing import List
import time
import re
from collections import Counter

from api_config import XIAOHU_MODEL_API
OPENAI_API_KEY = XIAOHU_MODEL_API["key"]
OPENAI_BASE_URL = XIAOHU_MODEL_API["url"]

class MedicalTermExtractor:
    def __init__(self, api_key=None, base_url=None, max_concurrent_requests=50):
        """
        Initialize medical term extractor
        
        Args:
            api_key: OpenAI API key
            base_url: API base URL (optional, for other compatible API services)
            max_concurrent_requests: Maximum number of concurrent API requests
        """
        self.client = AsyncOpenAI(
            base_url=OPENAI_BASE_URL, 
            api_key=OPENAI_API_KEY
        )
        self.extracted_terms = set()
        self.term_frequencies = Counter()
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
    async def extract_terms_with_llm(self, text: str, max_retries: int = 3) -> List[str]:
        """
        Extract medical terms using LLM with semaphore for rate limiting
        
        Args:
            text: Text to analyze
            max_retries: Maximum retry attempts
            
        Returns:
            List of extracted medical terms
        """
        ### 这个严格一点，一万多条乳腺的数据才提取出来50个左右的词组
        # prompt = f"""
        # Extract medical terms from the following ultrasound medical report description that belong to the following categories:
        # - Anatomical structures (e.g., heart, liver, fetal head circumference)
        # - Medical measurement indicators (e.g., biparietal diameter, abdominal circumference, femur length)
        # - Pathological states or findings (e.g., effusion, cyst, lesion, abnormality, calcification)
        # - Medical examination techniques or modalities (e.g., Doppler, color ultrasound, B-scan)
        # - Fetal development-related descriptors (e.g., gestational age, fetal position, intrauterine growth restriction)

        # Instructions:
        # - Extract terms exactly as they appear in the text, including multi-word phrases (e.g., 'fetal head circumference').
        # - Only include professional medical terms explicitly mentioned in the text; do not infer or add terms.
        # - Exclude:
        # - Measurement values (e.g., 3.4 cm, 120 bpm)
        # - Common adjectives (e.g., normal, mild) unless part of a recognized medical term (e.g., 'mild effusion' should only include 'effusion')
        # - Verbs or general clinical expressions (e.g., 'observed', 'seen')
        # - Patient demographic information (e.g., age, gender)

        # Output Format:
        # Return the output in JSON format with a single key 'medical_terms' containing a list of unique extracted terms, like this:
        # {{"medical_terms": ["term1", "term2", "term3"]}}

        # Text content:
        # {text}
        # """

        # prompt = f"""
        # Extract *all* professional medical terms from the following ultrasound report.
        # - Perform extraction case-insensitively and normalize hyphens/spaces.
        # - Include multi-word phrases (e.g. “mild effusion”, “left atrial enlargement”).
        # - Do not include numeric values (e.g. “3.4 cm”), verbs (“seen”), or patient demographics.
        # - Output as JSON: { "medical_terms": ["term1", "term2", …] }

        # Example:
        # Text: "There is a mild effusion around the heart and the biparietal diameter is 8.5 cm."
        # Output: {{ "medical_terms": ["mild effusion", "heart", "biparietal diameter"] }}

        # Text:
        # {text}
        # """
        prompt = f"""
        Extract *all* professional medical terms from the following ultrasound report.
        - Perform extraction case-insensitively and normalize hyphens/spaces.
        - Include multi-word phrases (e.g. "mild effusion", "left atrial enlargement").
        - Do not include numeric values (e.g. "3.4 cm"), verbs ("seen"), or patient demographics.
        - Output as JSON: {{"medical_terms": ["term1", "term2", "term3"]}}

        Example:
        Text: "There is a mild effusion around the heart and the biparietal diameter is 8.5 cm."
        Output: {{"medical_terms": ["mild effusion", "heart", "biparietal diameter"]}}

        Text:
        {text}
        """
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a professional medical term extraction expert specializing in extracting medical terminology from ultrasound medical reports."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content.strip()
                
                # Try to parse JSON
                try:
                    result = json.loads(content)
                    return result.get("medical_terms", [])
                except json.JSONDecodeError:
                    # If JSON parsing fails, try regex extraction
                    terms = re.findall(r'"([^"]+)"', content)
                    return [term for term in terms if len(term) > 1]
                    
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Extraction failed, skipping text: {text[:50]}...")
                    return []
        
        return []

    async def process_caption_batch(self, captions_batch: List[tuple], pbar=None):
        """
        Process a batch of captions concurrently
        
        Args:
            captions_batch: List of (sample_id, caption) tuples
            pbar: Progress bar for updating
        """
        # Create tasks for concurrent processing
        tasks = []
        for sample_id, caption in captions_batch:
            task = self.extract_terms_with_llm(caption)
            tasks.append((sample_id, task))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Process results
        for i, (sample_id, _) in enumerate(tasks):
            result = results[i]
            if isinstance(result, Exception):
                print(f"Error processing caption {sample_id}: {result}")
                continue
                
            terms = result if isinstance(result, list) else []
            
            # Add to set (automatic deduplication)
            for term in terms:
                if term and len(term.strip()) > 1:
                    clean_term = term.strip()
                    self.extracted_terms.add(clean_term)
                    self.term_frequencies[clean_term] += 1
            
            if pbar:
                pbar.update(1)
    
    async def process_json_files(self, data_dir: str, max_files: int = None, batch_size: int = 20):
        """
        Process all JSON files in the specified directory with concurrent caption processing
        
        Args:
            data_dir: Data directory path
            max_files: Maximum number of files to process (None for all files)
            batch_size: Number of captions to process concurrently in each batch
        """
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        json_files.sort()
        
        if max_files:
            json_files = json_files[:max_files]
        
        print(f"Found {len(json_files)} JSON files")
        print(f"Processing captions in batches of {batch_size}")
        
        total_captions = 0
        
        for json_file in tqdm(json_files, desc="Processing JSON files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                data_info = data.get('DataInfo', {})
                print(f"\nProcessing file: {os.path.basename(json_file)}")
                print(f"Contains {len(data_info)} entries")
                
                # Collect all captions from this file
                captions = []
                for sample_id, sample_data in data_info.items():
                    refined_caption = sample_data.get('CLIPcaption')
                    if refined_caption:
                        captions.append((sample_id, refined_caption))
                
                total_captions += len(captions)
                
                # Process captions in batches
                with tqdm(total=len(captions), desc="Extracting medical terms", leave=False) as pbar:
                    for i in range(0, len(captions), batch_size):
                        batch = captions[i:i + batch_size]
                        await self.process_caption_batch(batch, pbar)
                        
                        # Show progress every few batches
                        if (i // batch_size + 1) % 5 == 0:
                            print(f"Processed {min(i + batch_size, len(captions))}/{len(captions)} captions, "
                                  f"extracted {len(self.extracted_terms)} unique terms")
                    
            except Exception as e:
                print(f"Error processing file {json_file}: {e}")
                continue
        
        print(f"\nProcessing complete!")
        print(f"Total processed {total_captions} refined_captions")
        print(f"Extracted {len(self.extracted_terms)} unique medical terms")
    
    def save_vocabulary(self, output_path: str, min_frequency: int = 1):
        """
        Save vocabulary to file
        
        Args:
            output_path: Output file path
            min_frequency: Minimum frequency threshold (filter low-frequency words)
        """
        # Filter low-frequency words
        filtered_terms = [term for term, freq in self.term_frequencies.items() 
                         if freq >= min_frequency]
        
        # Sort by frequency (descending)
        filtered_terms.sort(key=lambda x: self.term_frequencies[x], reverse=True)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for term in filtered_terms:
                f.write(f"{term}\n")
        
        # Also save version with frequency
        freq_output_path = output_path.replace('.txt', '_with_frequency.txt')
        with open(freq_output_path, 'w', encoding='utf-8') as f:
            for term in filtered_terms:
                frequency = self.term_frequencies[term]
                f.write(f"{term}\t{frequency}\n")
        
        print(f"Vocabulary saved to: {output_path}")
        print(f"Vocabulary with frequency saved to: {freq_output_path}")
        print(f"Total saved {len(filtered_terms)} terms")

def main():
    """
    Main function
    """
    # Configuration parameters
    data_dir = "/media/ps/data-ssd/UltrasoundRAG/clip_caption/caption_breast"
    output_dir = "/media/ps/data-ssd/UltrasoundRAG/clip_caption/caption_breast/breast_vocab"
    vocab_file = os.path.join(output_dir, "breast_vocab_2.txt")
    
    # Concurrency settings
    max_concurrent_requests = 40  # Adjust based on your API rate limits
    batch_size = 20  # Number of captions to process in each batch
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory does not exist: {data_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize extractor with concurrency settings
    extractor = MedicalTermExtractor(max_concurrent_requests=max_concurrent_requests)
    print(f"\nStarting to process data directory: {data_dir}")
    print(f"Using {max_concurrent_requests} concurrent requests with batch size {batch_size}")
    
    # Start extraction
    try:
        asyncio.run(extractor.process_json_files(data_dir, batch_size=batch_size))
        # Save results
        min_freq = int(input("Set minimum frequency threshold (default 1): ").strip() or "1")
        extractor.save_vocabulary(vocab_file, min_frequency=min_freq)
        print("\nExtraction complete!")
    except KeyboardInterrupt:
        print("\nUser interrupted the process")
        if len(extractor.extracted_terms) > 0:
            print("Saving extracted terms...")
            extractor.save_vocabulary(vocab_file)
    except Exception as e:
        print(f"Error during processing: {e}")
        if len(extractor.extracted_terms) > 0:
            print("Saving extracted terms...")
            extractor.save_vocabulary(vocab_file)

if __name__ == "__main__":
    main()
