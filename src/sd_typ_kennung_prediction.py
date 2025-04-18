import os
from typing import Literal

import dspy
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pyprojroot import here
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


class SdTypKennung(dspy.Signature):
    text: str = dspy.InputField(desc="Versicherungsschadenmeldung")
    result: Literal[
        "Teilkasko",
        "Vollkasko",
        "Glass",
        "Einbruch / Diebstahl",
        "Elementar",
        "Feuer",
        "Leitungswasser",
        "Sturm",
        "Allgemeine Haftpflicht",
        "Haeuslicher Bereich",
        "Handyschaden",
        "Brillenschaden",
        "Kraftfahrthaftpflicht",
        "Sonstiges",
    ] = dspy.OutputField(
        desc="Wähle die Kategorie, die am besten zum Schaden basierend auf "
        "dem bereitgestellten Dokument passt. Wähle 'Sonstiges' nur "
        "dann, falls der Schaden keiner der anderen Kategorien "
        "zugeordnet werden kann."
    )


class Mapper:
    typ_mapping = {
        "TK": "Teilkasko",
        "VK": "Vollkasko",
        "GL": "Glass",
        "ED": "Einbruch / Diebstahl",
        "EL": "Elementar",
        "FE": "Feuer",
        "LW": "Leitungswasser",
        "ST": "Sturm",
        "AH": "Allgemeine Haftpflicht",
        "HB": "Haeuslicher Bereich",
        "HY": "Handyschaden",
        "BR": "Brillenschaden",
        "KH": "Kraftfahrthaftpflicht",
        "Other": "Sonstiges",
    }

    object_mapping = {
        "HR": "Hausrat",
        "WG": "Wohngebäude",
        "KF": "Kasko",
        "KH": "Kraftfahrthaftpflicht",
        "AH": "Allgemeine Haftpflicht",
        "GL": "Glass",
        "Other": "Sonstiges",
    }

    @classmethod
    def get_typ_full_name(cls, abbreviation):
        return cls.typ_mapping.get(abbreviation, "Unknown")

    @classmethod
    def get_typ_abbreviation(cls, full_name):
        reverse_typ_mapping = {v: k for k, v in cls.typ_mapping.items()}
        return reverse_typ_mapping.get(full_name, "Unknown")

    @classmethod
    def get_object_full_name(cls, abbreviation):
        return cls.object_mapping.get(abbreviation, "Unknown")

    @classmethod
    def get_object_abbreviation(cls, full_name):
        reverse_object_mapping = {v: k for k, v in cls.object_mapping.items()}
        return reverse_object_mapping.get(full_name, "Unknown")


if __name__ == "__main__":
    logger.info("Loading data")
    df = shuffle(pd.read_csv(here("./df_sample.csv")), random_state=42)

    logger.info("Selecting train & test for zero-shot learning")
    train = df.head(100)
    test = df.tail(20)

    logger.info("Loading OPENAI API KEY")
    _ = load_dotenv()

    logger.info("Setting dspy model")
    lm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    dspy.configure(lm=lm)

    logger.info("Zero-shot predictions for sd-typ-kennung")
    zero_shot_typ_predictor = dspy.Predict(SdTypKennung)
    typ_predictions = [
        zero_shot_typ_predictor(text=text) for text in test["anonymized_text"]
    ]

    logger.info("Mapping predictions to abbreviations for typ")
    typ_predictions = [
        Mapper.get_typ_abbreviation(pred.result) for pred in typ_predictions
    ]

    typ_accuracy = accuracy_score(
        test["sd_typ_kennung"], [pred for pred in typ_predictions]
    )
    logger.info(f"SD-Typ-Kennung accuracy: {typ_accuracy}")

    logger.info("Chain-of-Thought predictions for sd-typ-kennung")
    cot_typ_predictor = dspy.ChainOfThought(SdTypKennung)
    typ_predictions = [
        cot_typ_predictor(text=text) for text in test["anonymized_text"]
    ]

    logger.info("Mapping predictions to abbreviations for typ")
    typ_predictions = [
        Mapper.get_typ_abbreviation(pred.result) for pred in typ_predictions
    ]

    logger.info("Calculating accuracy for CoT SD-Typ-Kennung")
    typ_accuracy = accuracy_score(
        test["sd_typ_kennung"], [pred for pred in typ_predictions]
    )
    logger.info(f"CoT SD-Typ-Kennung accuracy: {typ_accuracy}")
