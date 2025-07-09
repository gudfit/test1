# src/data/lama_probe_loader.py
"""Load factual probes from LAMA dataset or similar knowledge probe datasets."""

import json
import random
import requests
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class LAMAProbeLoader:
    def __init__(self, cache_dir: str = "./data/lama_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.supported_relations = {
            "P19": "place of birth",
            "P20": "place of death",
            "P279": "subclass of",
            "P31": "instance of",
            "P36": "capital",
            "P37": "official language",
            "P47": "shares border with",
            "P101": "field of work",
            "P103": "native language",
            "P106": "occupation",
            "P108": "employer",
            "P127": "owned by",
            "P131": "located in",
            "P138": "named after",
            "P159": "headquarters location",
            "P176": "manufacturer",
            "P178": "developer",
            "P190": "sister city",
            "P264": "record label",
            "P276": "location",
            "P279": "subclass of",
            "P361": "part of",
            "P364": "original language",
            "P407": "language",
            "P413": "position played",
            "P449": "original network",
            "P495": "country of origin",
            "P527": "has part",
        }

    def load_t_rex_probes(self, num_probes: int = 1000) -> List[Dict[str, str]]:
        probes = []
        cache_file = self.cache_dir / "t_rex_probes.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                all_probes = json.load(f)
                random.shuffle(all_probes)
                return all_probes[:num_probes]
        probes = self._generate_t_rex_style_probes()
        with open(cache_file, "w") as f:
            json.dump(probes, f)

        random.shuffle(probes)
        return probes[:num_probes]

    def _generate_t_rex_style_probes(self) -> List[Dict[str, str]]:
        probes = []
        capitals = [
            ("France", "Paris"),
            ("Germany", "Berlin"),
            ("Italy", "Rome"),
            ("Spain", "Madrid"),
            ("Japan", "Tokyo"),
            ("China", "Beijing"),
            ("Russia", "Moscow"),
            ("Canada", "Ottawa"),
            ("Australia", "Canberra"),
            ("Brazil", "Brasília"),
            ("India", "Delhi"),
            ("Mexico", "Mexico"),
            ("Argentina", "Buenos Aires"),
            ("Egypt", "Cairo"),
            ("Turkey", "Ankara"),
            ("Greece", "Athens"),
            ("Netherlands", "Amsterdam"),
            ("Belgium", "Brussels"),
            ("Sweden", "Stockholm"),
            ("Norway", "Oslo"),
            ("Denmark", "Copenhagen"),
            ("Finland", "Helsinki"),
            ("Poland", "Warsaw"),
            ("Czech Republic", "Prague"),
            ("Austria", "Vienna"),
            ("Switzerland", "Bern"),
            ("Portugal", "Lisbon"),
            ("Ireland", "Dublin"),
            ("Scotland", "Edinburgh"),
            ("Wales", "Cardiff"),
        ]

        for country, capital in capitals:
            probes.append(
                {
                    "prompt": f"The capital of {country} is [MASK]",
                    "answer": capital,
                    "relation": "P36",
                }
            )

        languages = [
            ("France", "French"),
            ("Germany", "German"),
            ("Spain", "Spanish"),
            ("Italy", "Italian"),
            ("Japan", "Japanese"),
            ("China", "Chinese"),
            ("Russia", "Russian"),
            ("Brazil", "Portuguese"),
            ("Netherlands", "Dutch"),
            ("Sweden", "Swedish"),
            ("Norway", "Norwegian"),
            ("Denmark", "Danish"),
            ("Finland", "Finnish"),
            ("Poland", "Polish"),
            ("Greece", "Greek"),
            ("Turkey", "Turkish"),
            ("India", "Hindi"),
            ("Egypt", "Arabic"),
        ]

        for country, language in languages:
            probes.append(
                {
                    "prompt": f"The official language of {country} is [MASK]",
                    "answer": language,
                    "relation": "P37",
                }
            )

        instance_relations = [
            ("Paris", "city"),
            ("Tokyo", "city"),
            ("Amazon", "river"),
            ("Everest", "mountain"),
            ("Pacific", "ocean"),
            ("Sahara", "desert"),
            ("Europe", "continent"),
            ("Asia", "continent"),
            ("Mars", "planet"),
            ("Jupiter", "planet"),
            ("Sun", "star"),
            ("Moon", "satellite"),
            ("Apple", "company"),
            ("Google", "company"),
            ("Harvard", "university"),
            ("Oxford", "university"),
            ("Python", "language"),
            ("Java", "language"),
        ]

        for instance, type_of in instance_relations:
            probes.append(
                {
                    "prompt": f"{instance} is a [MASK]",
                    "answer": type_of,
                    "relation": "P31",
                }
            )

        locations = [
            ("Eiffel Tower", "Paris"),
            ("Statue of Liberty", "New York"),
            ("Big Ben", "London"),
            ("Colosseum", "Rome"),
            ("Taj Mahal", "Agra"),
            ("Great Wall", "China"),
            ("Pyramids", "Egypt"),
            ("Acropolis", "Athens"),
            ("Kremlin", "Moscow"),
            ("White House", "Washington"),
            ("Buckingham Palace", "London"),
            ("Louvre", "Paris"),
            ("Vatican", "Rome"),
            ("Brandenburg Gate", "Berlin"),
        ]

        for landmark, location in locations:
            probes.append(
                {
                    "prompt": f"The {landmark} is located in [MASK]",
                    "answer": location,
                    "relation": "P276",
                }
            )

        occupations = [
            ("Einstein", "physicist"),
            ("Darwin", "biologist"),
            ("Shakespeare", "playwright"),
            ("Mozart", "composer"),
            ("Picasso", "painter"),
            ("Newton", "physicist"),
            ("Beethoven", "composer"),
            ("Leonardo da Vinci", "artist"),
            ("Galileo", "astronomer"),
            ("Marie Curie", "chemist"),
            ("Tesla", "inventor"),
            ("Edison", "inventor"),
            ("Plato", "philosopher"),
            ("Aristotle", "philosopher"),
        ]

        for person, occupation in occupations:
            probes.append(
                {
                    "prompt": f"{person} was a [MASK]",
                    "answer": occupation,
                    "relation": "P106",
                }
            )
        companies = [
            ("Microsoft", "Windows"),
            ("Apple", "iPhone"),
            ("Google", "Search"),
            ("Amazon", "AWS"),
            ("Tesla", "cars"),
            ("Boeing", "aircraft"),
            ("Nike", "shoes"),
            ("McDonald's", "burgers"),
            ("Coca-Cola", "soda"),
            ("Toyota", "cars"),
            ("Samsung", "phones"),
            ("Sony", "PlayStation"),
        ]

        for company, product in companies:
            probes.append(
                {
                    "prompt": f"{company} makes [MASK]",
                    "answer": product,
                    "relation": "P176",
                }
            )

        return probes

    def load_conceptnet_probes(self, num_probes: int = 1000) -> List[Dict[str, str]]:
        probes = []
        conceptnet_data = [
            ("dog", "animal"),
            ("cat", "animal"),
            ("car", "vehicle"),
            ("airplane", "vehicle"),
            ("apple", "fruit"),
            ("banana", "fruit"),
            ("carrot", "vegetable"),
            ("potato", "vegetable"),
            ("rose", "flower"),
            ("oak", "tree"),
            ("salmon", "fish"),
            ("eagle", "bird"),
            ("python", "snake"),
            ("violin", "instrument"),
            ("piano", "instrument"),
            ("wheel", "car"),
            ("wing", "airplane"),
            ("leaf", "tree"),
            ("petal", "flower"),
            ("chapter", "book"),
            ("scene", "movie"),
            ("verse", "song"),
            ("ingredient", "recipe"),
            ("player", "team"),
            ("pen", "writing"),
            ("knife", "cutting"),
            ("car", "transportation"),
            ("phone", "communication"),
            ("book", "reading"),
            ("bed", "sleeping"),
            ("stove", "cooking"),
            ("camera", "photography"),
            ("telescope", "astronomy"),
            ("ice", "cold"),
            ("fire", "hot"),
            ("sugar", "sweet"),
            ("lemon", "sour"),
            ("rock", "hard"),
            ("pillow", "soft"),
            ("gold", "valuable"),
            ("diamond", "expensive"),
            ("feather", "light"),
        ]

        for item1, item2 in conceptnet_data[: len(conceptnet_data) // 3]:
            probes.append(
                {
                    "prompt": f"A {item1} is a type of [MASK]",
                    "answer": item2,
                    "relation": "IsA",
                }
            )

        for item1, item2 in conceptnet_data[
            len(conceptnet_data) // 3 : 2 * len(conceptnet_data) // 3
        ]:
            probes.append(
                {
                    "prompt": f"A {item1} is part of a [MASK]",
                    "answer": item2,
                    "relation": "PartOf",
                }
            )

        for item1, item2 in conceptnet_data[2 * len(conceptnet_data) // 3 :]:
            probes.append(
                {
                    "prompt": f"A {item1} is [MASK]",
                    "answer": item2,
                    "relation": "HasProperty",
                }
            )

        random.shuffle(probes)
        return probes[:num_probes]

    def load_squad_based_probes(self, num_probes: int = 1000) -> List[Dict[str, str]]:
        probes = []
        historical = [
            ("America", "1492", "Columbus discovered"),
            ("World War II", "1945", "ended in"),
            ("World War I", "1918", "ended in"),
            ("American Independence", "1776", "declared in"),
            ("French Revolution", "1789", "began in"),
            ("Berlin Wall", "1989", "fell in"),
            ("Soviet Union", "1991", "dissolved in"),
            ("United Nations", "1945", "founded in"),
            ("NATO", "1949", "established in"),
            ("European Union", "1993", "formed in"),
        ]

        for event, year, context in historical:
            probes.append(
                {
                    "prompt": f"{context} {event} in [MASK]",
                    "answer": year,
                    "relation": "temporal",
                }
            )

        scientific = [
            ("water", "H2O", "chemical formula"),
            ("salt", "NaCl", "chemical formula"),
            ("photosynthesis", "oxygen", "produces"),
            ("gravity", "Newton", "discovered by"),
            ("evolution", "Darwin", "theory by"),
            ("relativity", "Einstein", "theory by"),
            ("DNA", "Watson", "structure discovered by"),
            ("penicillin", "Fleming", "discovered by"),
            ("radioactivity", "Curie", "studied by"),
        ]

        for concept, answer, context in scientific:
            if context == "chemical formula":
                prompt = f"The {context} for {concept} is [MASK]"
            elif context.endswith("by"):
                prompt = f"{concept.capitalize()} was {context} [MASK]"
            else:
                prompt = f"{concept.capitalize()} {context} [MASK]"

            probes.append(
                {"prompt": prompt, "answer": answer, "relation": "scientific"}
            )

        geographical = [
            ("Nile", "Africa", "longest river in"),
            ("Amazon", "South America", "largest river in"),
            ("Everest", "Asia", "highest mountain in"),
            ("Sahara", "Africa", "largest desert in"),
            ("Pacific", "ocean", "largest"),
            ("Russia", "country", "largest"),
            ("Vatican", "country", "smallest"),
            ("China", "population", "largest"),
            ("Tokyo", "city", "largest"),
        ]

        for feature, location, context in geographical:
            if context.endswith("in"):
                prompt = f"The {feature} is the {context} [MASK]"
            else:
                prompt = f"{feature} is the {context} [MASK]"

            probes.append(
                {"prompt": prompt, "answer": location, "relation": "geographical"}
            )

        random.shuffle(probes)
        return probes[:num_probes]

    def create_mixed_probe_set(self, num_probes: int = 1000) -> List[Dict[str, str]]:
        probes = []
        n_per_source = num_probes // 3
        probes.extend(self.load_t_rex_probes(n_per_source))
        probes.extend(self.load_conceptnet_probes(n_per_source))
        probes.extend(self.load_squad_based_probes(num_probes - 2 * n_per_source))
        random.shuffle(probes)
        return probes[:num_probes]

    def filter_probes_by_model_type(
        self, probes: List[Dict[str, str]], model_type: str
    ) -> List[Dict[str, str]]:
        filtered = []

        for probe in probes:
            prompt = probe["prompt"]
            answer = probe["answer"]

            if model_type in ["bert", "distilbert"]:
                if len(answer.split()) == 1:
                    filtered.append(probe)
            else:
                filtered.append(probe)

        return filtered
