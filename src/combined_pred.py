import os
from typing import Literal

from dotenv import load_dotenv
from dspy import Signature, InputField, OutputField, Predict, LM, configure
from loguru import logger
from pydantic import BaseModel, Field


class BasisSchaden(BaseModel):
    typ: Literal[
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
    ] = Field(
        ...,
        description="Wähle die Kategorie, die am besten zum Schaden basierend auf "
        "dem bereitgestellten Dokument passt. Wähle 'Sonstiges' nur "
        "dann, falls der Schaden keiner der anderen Kategorien "
        "zugeordnet werden kann.",
    )
    objekt: Literal[
        "Hausrat",
        "Wohngebäude",
        "Kasko",
        "Kraftfahrthaftpflicht",
        "Allgemeine Haftpflicht",
        "Glass",
        "Sonstiges",
    ] = Field(
        ...,
        description="Wähle die Kategorie, die am besten zum beschädigten Objekt "
        "basierend auf dem bereitgestellten Dokument passt. Wähle "
        "'Sonstiges', falls keine der anderen Kategorien passt.",
    )


class FNOL(Signature):
    text: str = InputField(desc="Versicherungsschadenmeldung")
    result: BasisSchaden = OutputField(
        desc="Kategorisierung des Schadens und des beschädigten Objekts"
    )


if __name__ == "__main__":
    logger.info("Loading OPENAI API KEY")
    _ = load_dotenv()

    logger.info("Setting dspy model")
    lm = LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    configure(lm=lm)

    fnol_predictor = Predict(FNOL)

    sample_text = "Der Versicherungsnehmer meldet einen Schaden am 01.01.2023. Ihm wurde sein Fahrrad gestohlen."

    result = fnol_predictor(text=sample_text).toDict()
    logger.info(result)

    logger.success("Prediction finished")
