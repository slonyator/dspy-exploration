from typing import Literal, Any

import instructor
import pandas as pd
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator
from pyprojroot import here
from sklearn.utils import shuffle


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


def validate_combination(typ, objekt) -> bool:
    """
    Validates if the combination of typ and objekt is valid.
    Returns True if the combination is valid, otherwise False.
    Maps the full names to abbreviations and checks if the combination is valid.
    """
    logger.info("Entering validation function")
    try:
        logger.info("Mapping schaden-objekt from full name to abbreviation")
        objekt_abbr = Mapper.get_object_abbreviation(objekt)
        logger.info(f"Mapped object: {objekt_abbr}")

        logger.info("Mapping schaden-typ from full name to abbreviation")
        typ_abbr = Mapper.get_typ_abbreviation(typ)
        logger.info(f"Mapped type: {typ_abbr}")

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

        return (objekt_abbr, typ_abbr) in valid_combinations
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        return False


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

    @model_validator(mode="before")
    @classmethod
    def validate_prediction(cls, data: dict[str, Any]) -> dict[str, Any]:
        # Normalize the input: if the value is already an abbreviation, use it;
        # otherwise convert from full name.
        typ_value = data.get("typ")
        objekt_value = data.get("objekt")
        if typ_value in Mapper.typ_mapping:  # already abbreviation
            typ_abbr = typ_value
        else:
            typ_abbr = Mapper.get_typ_abbreviation(typ_value)
        if objekt_value in Mapper.object_mapping:
            objekt_abbr = objekt_value
        else:
            objekt_abbr = Mapper.get_object_abbreviation(objekt_value)

        # Define the allowed (objekt_abbr, typ_abbr) combinations.
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
        if (objekt_abbr, typ_abbr) not in valid_combinations:
            raise ValueError("Invalid combination of typ and objekt")
        logger.info("Valid combination of typ and objekt")
        logger.info("Mapping objekt and typ to full names")
        # Convert abbreviations back to full names.
        data["typ"] = Mapper.get_typ_full_name(typ_abbr)
        data["objekt"] = Mapper.get_object_full_name(objekt_abbr)
        return data


def get_prediction(client: instructor.client, doc: str) -> BasisSchaden | None:
    try:
        pred = client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=BasisSchaden,
            temperature=0,
            max_retries=2,
            messages=[
                {"role": "user", "content": simple_prompt},
                {"role": "user", "content": doc},
            ],
        )

        return pred
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None


if __name__ == "__main__":
    logger.info("Loading OPENAI API KEY")
    _ = load_dotenv()

    logger.info("Loading sample data")
    df = shuffle(
        pd.read_csv(here("./df_sample.csv")).filter(
            ["anonymized_text", "schaden_objekt", "sd_typ_kennung"]
        ),
        random_state=42,
    )

    testset = df.tail(200).sample(10, random_state=42).reset_index(drop=True)

    logger.info("Loading simple prompt from jinja template")
    env = Environment(loader=FileSystemLoader(here("./src/")))
    template = env.get_template("simple_prompt.jinja")
    simple_prompt = template.render()

    logger.info("Setting instructor client")
    client = instructor.from_openai(OpenAI())

    logger.info("Predicting on testset")
    predictions = [
        get_prediction(client=client, doc=doc)
        for doc in testset["anonymized_text"]
    ]

    logger.info("Evaluating predictions on the true labels")
    correct_predictions = sum(
        1
        for i, pred in enumerate(predictions)
        if pred is not None
        and pred.typ
        == Mapper.get_typ_abbreviation(testset.iloc[i]["sd_typ_kennung"])
        and pred.objekt
        == Mapper.get_object_abbreviation(testset.iloc[i]["schaden_objekt"])
    )

    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions
    logger.info(f"Accuracy: {accuracy:.2f}")
