import os

from dotenv import load_dotenv
from pyprojroot import here
from loguru import logger
from sklearn.utils import shuffle
from typing import Literal

import dspy
import pandas as pd


class SchadenObjekt(dspy.Signature):
    text: str = dspy.InputField(desc="Versicherungsschadenmeldung")

    result: Literal[
        "Hausrat",
        "Wohngebäude",
        "Kasko",
        "Kraftfahrthaftpflicht",
        "Allgemeine Haftpflicht",
        "Glass",
        "Sonstiges",
    ] = dspy.OutputField(
        desc="Wähle die Kategorie, die am besten zum beschädigten Objekt "
        "basierend auf dem bereitgestellten Dokument passt. Wähle "
        "'Sonstiges', falls keine der anderen Kategorien passt."
    )


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

    logger.info("Selecting testset for zero-shot learning")
    test = df.tail(20)

    logger.info("Loading OPENAI API KEY")

    _ = load_dotenv()

    logger.info("Setting dspy model")
    lm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    dspy.configure(lm=lm)

    logger.info("Zero-shot predictions for SchadenObjekt")
    zero_shot_objekt_predictor = dspy.Predict(SchadenObjekt)
    objekt_predictions = [
        zero_shot_objekt_predictor(text=text)
        for text in test["anonymized_text"]
    ]

    logger.info("Zero-shot predictions for SdTypKennung")
    zero_shot_typ_predictor = dspy.Predict(SdTypKennung)
    typ_predictions = [
        zero_shot_typ_predictor(text=text) for text in test["anonymized_text"]
    ]
