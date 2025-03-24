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
