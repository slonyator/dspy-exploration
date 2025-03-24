import os
from typing import Literal

from dotenv import load_dotenv
from dspy import Signature, InputField, OutputField, Predict, LM, configure
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


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

    @model_validator(mode="after")
    def check_valid_combination(self) -> Self:
        valid_combinations = {
            ("HR", "LW"),
            ("HR", "ST"),
            ("HR", "FE"),
            ("HR", "EL"),
            ("HR", "ED"),
            ("HR", "GL"),
            ("WG", "LW"),
            ("WG", "ST"),
            ("WG", "FE"),
            ("WG", "EL"),
            ("WG", "ED"),
            ("WG", "GL"),
            ("GL", "GL"),
            ("KF", "VK"),
            ("KF", "TK"),
            ("KH", "KH"),
            ("AH", "AH"),
            ("AH", "HB"),
            ("AH", "HY"),
            ("AH", "BR"),
        }

        logger.info("Mapping schaden-objekt from full name to abbreviation")
        self.objekt = Mapper.get_object_abbreviation(self.objekt)

        logger.info("Mapping schaden-typ from full name to abbreviation")
        self.typ = Mapper.get_typ_abbreviation(self.typ)

        if (self.objekt, self.typ) not in valid_combinations:
            raise ValueError("Kombination von Typ und Objekt ist nicht gültig")
        return Self


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
