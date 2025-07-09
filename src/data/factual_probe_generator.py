# src/data/factual_probe_generator.py
"""Generate factual probes from WikiText or other datasets."""

import re
import random
import spacy
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class FactualProbeGenerator:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading spaCy model {spacy_model}...")
            import subprocess
            subprocess.run([f"python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)
            
        # Patterns for extracting factual statements
        self.patterns = {
            'is_a': r'(\w+(?:\s+\w+)*)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+)*)',
            'located_in': r'(\w+(?:\s+\w+)*)\s+(?:in|at|on|near)\s+(\w+(?:\s+\w+)*)',
            'year_fact': r'(?:In|During|Around)?\s*(\d{4})\s*,?\s*(.+?)(?:\.|,|;)',
            'numerical_fact': r'(\w+(?:\s+\w+)*)\s+(?:has|have|had|contains?)\s+(\d+)\s+(\w+)',
            'definition': r'(\w+(?:\s+\w+)*)\s+(?:is defined as|means|refers to)\s+(.+?)(?:\.|,|;)',
        }
        
    def extract_from_wikitext(self, num_probes: int = 1000, 
                            dataset_split: str = 'train',
                            max_samples: int = 5000) -> List[Dict[str, str]]:
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=dataset_split)
        
        probes = []
        processed_samples = 0
        
        for item in dataset:
            if processed_samples >= max_samples:
                break
                
            text = item['text'].strip()
            if len(text) < 50:  
                continue
            text_probes = self._extract_probes_from_text(text)
            probes.extend(text_probes)
            processed_samples += 1
            if len(probes) >= num_probes:
                break
                
        unique_probes = self._deduplicate_probes(probes)
        random.shuffle(unique_probes)
        
        return unique_probes[:num_probes]
    
    def _extract_probes_from_text(self, text: str) -> List[Dict[str, str]]:
        probes = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            if len(sentence.split()) < 5 or len(sentence.split()) > 30:
                continue
                
            probes.extend(self._extract_entity_facts(sentence))
            probes.extend(self._extract_numerical_facts(sentence))
            probes.extend(self._extract_definition_facts(sentence))
            probes.extend(self._extract_relation_facts(sentence))
            
        return probes
    
    def _extract_entity_facts(self, sentence: str) -> List[Dict[str, str]]:
        probes = []
        doc = self.nlp(sentence)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        for ent_text, ent_label in entities:
            if len(ent_text.split()) < 1:
                continue
                
            if ent_label == "PERSON":
                if "born" in sentence.lower():
                    probe = self._create_birth_probe(sentence, ent_text)
                    if probe:
                        probes.append(probe)
                elif any(word in sentence.lower() for word in ["wrote", "created", "invented", "discovered"]):
                    probe = self._create_creation_probe(sentence, ent_text)
                    if probe:
                        probes.append(probe)
                        
            elif ent_label in ["GPE", "LOC"]:  
                if "capital" in sentence.lower():
                    probe = self._create_capital_probe(sentence, ent_text)
                    if probe:
                        probes.append(probe)
                        
            elif ent_label == "DATE":
                probe = self._create_date_probe(sentence, ent_text)
                if probe:
                    probes.append(probe)
                    
        return probes
    
    def _extract_numerical_facts(self, sentence: str) -> List[Dict[str, str]]:
        probes = []
        
        num_pattern = r'(\w+(?:\s+\w+)*)\s+(?:has|have|had|contains?|is|are|was|were)\s+(\d+(?:,\d+)*(?:\.\d+)?)\s+(\w+)'
        matches = re.finditer(num_pattern, sentence, re.IGNORECASE)
        
        for match in matches:
            subject = match.group(1)
            number = match.group(2)
            unit = match.group(3)
            
            prompt = f"{subject} has [MASK] {unit}"
            answer = number.replace(",", "")
            
            if len(subject.split()) < 10 and len(answer) < 20:
                probes.append({
                    "prompt": prompt,
                    "answer": answer,
                    "source_sentence": sentence
                })
                
        return probes
    
    def _extract_definition_facts(self, sentence: str) -> List[Dict[str, str]]:
        probes = []
        
        def_patterns = [
            r'(\w+(?:\s+\w+)*)\s+is\s+(?:a|an|the)\s+(\w+(?:\s+\w+){0,3})',
            r'(\w+(?:\s+\w+)*),\s+(?:a|an|the)\s+(\w+(?:\s+\w+){0,3})',
        ]
        
        for pattern in def_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            
            for match in matches:
                subject = match.group(1)
                definition = match.group(2)
                if len(subject.split()) > 5 or len(definition.split()) > 5:
                    continue
                prompt = f"{subject} is a [MASK]"
                answer = definition.split()[-1]  
                
                if answer.isalpha() and len(answer) > 2:
                    probes.append({
                        "prompt": prompt,
                        "answer": answer.lower(),
                        "source_sentence": sentence
                    })
                    
        return probes
    
    def _extract_relation_facts(self, sentence: str) -> List[Dict[str, str]]:
        probes = []
        doc = self.nlp(sentence)
        
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                subject = None
                obj = None
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child.text
                        for subchild in child.children:
                            if subchild.dep_ == "compound":
                                subject = f"{subchild.text} {subject}"
                    elif child.dep_ in ["dobj", "pobj"]:
                        obj = child.text
                        for subchild in child.children:
                            if subchild.dep_ == "compound":
                                obj = f"{subchild.text} {obj}"
                                
                if subject and obj and len(subject.split()) < 5 and len(obj.split()) < 5:
                    prompt = f"{subject} {token.text} [MASK]"
                    answer = obj
                    
                    probes.append({
                        "prompt": prompt,
                        "answer": answer.lower(),
                        "source_sentence": sentence
                    })
                    
        return probes
    
    def _create_birth_probe(self, sentence: str, person: str) -> Optional[Dict[str, str]]:
        year_match = re.search(r'born\s+(?:in\s+)?(\d{4})', sentence, re.IGNORECASE)
        if year_match:
            return {
                "prompt": f"{person} was born in [MASK]",
                "answer": year_match.group(1),
                "source_sentence": sentence
            }
        return None
    
    def _create_creation_probe(self, sentence: str, person: str) -> Optional[Dict[str, str]]:
        """Create probe for creation/invention facts."""
        patterns = [
            (r'wrote\s+(?:the\s+)?([A-Z][^,\.]+)', "wrote"),
            (r'invented\s+(?:the\s+)?([A-Z][^,\.]+)', "invented"),
            (r'discovered\s+(?:the\s+)?([A-Z][^,\.]+)', "discovered"),
            (r'created\s+(?:the\s+)?([A-Z][^,\.]+)', "created")
        ]
        
        for pattern, verb in patterns:
            match = re.search(pattern, sentence)
            if match and person in sentence:
                creation = match.group(1).strip()
                if len(creation.split()) < 10:
                    return {
                        "prompt": f"{person} {verb} [MASK]",
                        "answer": creation.split()[-1],
                        "source_sentence": sentence
                    }
        return None
    
    def _create_capital_probe(self, sentence: str, location: str) -> Optional[Dict[str, str]]:
        if "capital of" in sentence.lower():
            match = re.search(r'capital\s+of\s+(\w+(?:\s+\w+)*)', sentence, re.IGNORECASE)
            if match:
                country = match.group(1)
                return {
                    "prompt": f"The capital of {country} is [MASK]",
                    "answer": location.split()[-1],  
                    "source_sentence": sentence
                }
        return None
    
    def _create_date_probe(self, sentence: str, date: str) -> Optional[Dict[str, str]]:
        doc = self.nlp(sentence)
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                subject = None
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child.text
                        break
                        
                if subject and len(subject.split()) < 5:
                    return {
                        "prompt": f"{subject} happened in [MASK]",
                        "answer": date.split()[-1],  
                        "source_sentence": sentence
                    }
        return None
    
    def _deduplicate_probes(self, probes: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen_prompts = set()
        unique_probes = []
        
        for probe in probes:
            normalized_prompt = probe['prompt'].lower().strip()
            
            is_duplicate = False
            for seen in seen_prompts:
                if self._are_similar(normalized_prompt, seen):
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                seen_prompts.add(normalized_prompt)
                unique_probes.append(probe)
                
        return unique_probes
    
    def _are_similar(self, prompt1: str, prompt2: str, threshold: float = 0.8) -> bool:
        """Check if two prompts are similar using simple overlap."""
        words1 = set(prompt1.split())
        words2 = set(prompt2.split())
        
        if not words1 or not words2:
            return False
            
        overlap = len(words1.intersection(words2))
        similarity = overlap / min(len(words1), len(words2))
        
        return similarity > threshold
    
    def generate_from_templates(self, entities: Dict[str, List[str]], 
                              num_probes: int = 1000) -> List[Dict[str, str]]:
        """
        Generate probes using templates and entity lists.
        
        Args:
            entities: Dictionary of entity types and their values
            num_probes: Number of probes to generate
            
        Returns:
            List of factual probes
        """
        templates = {
            'capitals': {
                'template': "The capital of {country} is [MASK]",
                'entities': 'countries',
                'answers': 'capitals'
            },
            'inventions': {
                'template': "{inventor} invented the [MASK]",
                'entities': 'inventors',
                'answers': 'inventions'
            },
            'years': {
                'template': "{event} occurred in [MASK]",
                'entities': 'events',
                'answers': 'years'
            }
        }
        
        probes = []
        
        for template_type, template_info in templates.items():
            if template_info['entities'] in entities and template_info['answers'] in entities:
                entity_list = entities[template_info['entities']]
                answer_list = entities[template_info['answers']]
                
                for entity, answer in zip(entity_list, answer_list):
                    probe = {
                        'prompt': template_info['template'].format(**{template_info['entities'][:-1]: entity}),
                        'answer': answer
                    }
                    probes.append(probe)
                    
                    if len(probes) >= num_probes:
                        break
                        
        return probes[:num_probes]
