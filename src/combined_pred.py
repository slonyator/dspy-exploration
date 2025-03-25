import os
from typing import Literal

from dotenv import load_dotenv
from dspy import (
    BootstrapFewShot,
    Example,
    Evaluate,
    Signature,
    InputField,
    OutputField,
    Predict,
    LM,
    configure,
)
from dspy.predict.refine import Refine
from loguru import logger
from pydantic import BaseModel, Field
from pyprojroot import here
from sklearn.utils import shuffle

import pandas as pd


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


class FNOL(Signature):
    text: str = InputField(desc="Versicherungsschadenmeldung")
    result: BasisSchaden = OutputField(
        desc="Kategorisierung des Schadens und des beschädigten Objekts"
    )


def validate_combination(_args, pred) -> float:
    """
    Reward function for Refine.
    Returns 1.0 if the combination is valid, otherwise 0.0.
    Maps the full names to abbreviations and checks if the combination is valid.
    """
    logger.info("Entering validation function")
    try:
        basis_schaden = pred.result
        typ = basis_schaden.typ
        objekt = basis_schaden.objekt

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

        return 1.0 if (objekt_abbr, typ_abbr) in valid_combinations else 0.0
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        return 0.0


def result_exact_match(example, prediction, trace=None):
    logger.info(
        f"example: {example}, prediction: {prediction}, trace: {trace}"
    )
    return example.result == prediction.result


if __name__ == "__main__":
    logger.info("Loading OPENAI API KEY")
    _ = load_dotenv()

    logger.info("Setting dspy model")
    lm = LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    configure(lm=lm)

    predictor = Refine(
        module=Predict(FNOL),
        N=2,
        reward_fn=validate_combination,
        threshold=1.0,
    )

    sample_text = (
        "Der Versicherungsnehmer meldet einen Schaden am 01.01.2023. "
        "Ihm wurde sein Fahrrad gestohlen."
    )

    result = predictor(text=sample_text)
    logger.info(f"Final result: {result}")

    logger.info("Loading the full dataset")
    df = shuffle(
        pd.read_csv(here("./df_sample.csv")).filter(
            ["anonymized_text", "schaden_objekt", "sd_typ_kennung"]
        ),
        random_state=42,
    )

    trainset = [
        Example(
            text=x["anonymized_text"],
            result=BasisSchaden(
                typ=Mapper.get_typ_full_name(x["sd_typ_kennung"]),
                objekt=Mapper.get_object_full_name(x["schaden_objekt"]),
            ),
        ).with_inputs("text")
        for x in df.head(100).to_dict("records")
    ]
    testset = [
        Example(
            text=x["anonymized_text"],
            result=BasisSchaden(
                typ=Mapper.get_typ_full_name(x["sd_typ_kennung"]),
                objekt=Mapper.get_object_full_name(x["schaden_objekt"]),
            ),
        ).with_inputs("text")
        for x in df.tail(200).to_dict("records")
    ]

    logger.info("Evaluation with Zero-Shot-Predictions")

    evaluate_program = Evaluate(
        devset=testset,
        metric=result_exact_match,
        display_progress=True,
        display_table=10,
    )

    eval = evaluate_program(predictor)
    logger.info(f"Evaluation results: {eval}")

    logger.info("Setting up BootstrapFewShot teleprompter")
    teleprompter = BootstrapFewShot(
        metric=result_exact_match, max_labeled_demos=10
    )

    logger.info("Compiling predictor with few-shot learning")
    compiled_predictor = teleprompter.compile(predictor, trainset=trainset)

    logger.info("Evaluating compiled predictor on test set")
    evaluate_program = Evaluate(
        devset=testset,
        metric=result_exact_match,
        num_threads=1,
        display_progress=True,
        display_table=10,
    )

    logger.info("Run the evaluation after compilation")
    eval_compiled = evaluate_program(compiled_predictor)
    logger.info(f"Evaluation Result: {eval_compiled}")
