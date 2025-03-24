import os

from dotenv import load_dotenv
from dspy.evaluate.metrics import answer_exact_match
from dspy.evaluate import Evaluate
from pyprojroot import here
from loguru import logger
from sklearn.metrics import accuracy_score
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

    logger.info("Selecting train & test for zero-shot learning")
    train = df.head(100)
    test = df.tail(20)

    logger.info("Loading OPENAI API KEY")

    _ = load_dotenv()

    logger.info("Setting dspy model")
    lm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    dspy.configure(lm=lm)

    logger.info("Zero-shot predictions for schaden-objekt")
    zero_shot_objekt_predictor = dspy.Predict(SchadenObjekt)
    objekt_predictions = [
        zero_shot_objekt_predictor(text=text)
        for text in test["anonymized_text"]
    ]

    logger.info("Zero-shot predictions for sd-typ-kennung")
    zero_shot_typ_predictor = dspy.Predict(SdTypKennung)
    typ_predictions = [
        zero_shot_typ_predictor(text=text) for text in test["anonymized_text"]
    ]

    logger.info("Mapping predictions to abbreviations for object")
    objekt_predictions = [
        Mapper.get_object_abbreviation(pred.result)
        for pred in objekt_predictions
    ]

    logger.info("Mapping predictions to abbreviations for typ")
    typ_predictions = [
        Mapper.get_typ_abbreviation(pred.result) for pred in typ_predictions
    ]

    logger.info("Calculating accuracy")
    objekt_accuracy = accuracy_score(
        test["schaden_objekt"], [pred for pred in objekt_predictions]
    )

    logger.info(f"Schaden-Objekt accuracy: {objekt_accuracy}")

    typ_accuracy = accuracy_score(
        test["sd_typ_kennung"], [pred for pred in typ_predictions]
    )

    logger.info(f"SD-Typ-Kennung accuracy: {typ_accuracy}")

    logger.info("Chain-of-Thought predictions for schaden-objekt")
    cot_objekt_predictor = dspy.ChainOfThought(SchadenObjekt)
    objekt_predictions = [
        cot_objekt_predictor(text=text) for text in test["anonymized_text"]
    ]

    logger.info("Mapping predictions to abbreviations for object")
    objekt_predictions = [
        Mapper.get_object_abbreviation(pred.result)
        for pred in objekt_predictions
    ]

    logger.info("Calculating accuracy for CoT Schaden-Objekt")
    objekt_accuracy = accuracy_score(
        test["schaden_objekt"], [pred for pred in objekt_predictions]
    )

    logger.info(f"CoT Schaden-Objekt accuracy: {objekt_accuracy}")

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

    logger.info("Starting with FS Prompt optimization for Schaden-Objekt")
    # so_train = train.filter(["anonymized_text", "schaden_objekt"])
    # so_test = test.filter(["anonymized_text", "schaden_objekt"])
    #
    # logger.info("Mapping labels to full names")
    # so_train["schaden_objekt"] = so_train["schaden_objekt"].apply(Mapper.get_object_full_name)
    # so_test["schaden_objekt"] = so_test["schaden_objekt"].apply(Mapper.get_object_full_name)
    #
    # trainset = [
    #     dspy.Example(
    #         question=x["anonymized_text"], answer=x["schaden_objekt"]
    #     ).with_inputs("question")
    #     for x in so_train.to_dict("records")
    # ]
    # testset = [
    #     dspy.Example(
    #         question=x["anonymized_text"], answer=x["schaden_objekt"]
    #     ).with_inputs("question")
    #     for x in so_test.to_dict("records")
    # ]

    so_train = train.filter(items=["anonymized_text", "schaden_objekt"])
    so_test = test.filter(items=["anonymized_text", "schaden_objekt"])

    logger.info("Mapping labels to full names")
    so_train["schaden_objekt"] = so_train["schaden_objekt"].apply(
        Mapper.get_object_full_name
    )
    so_test["schaden_objekt"] = so_test["schaden_objekt"].apply(
        Mapper.get_object_full_name
    )

    trainset = [
        dspy.Example(
            anonymized_text=x["anonymized_text"],
            schaden_objekt=x["schaden_objekt"],
        ).with_inputs("anonymized_text")
        for x in so_train.to_dict("records")
    ]
    testset = [
        dspy.Example(
            anonymized_text=x["anonymized_text"],
            schaden_objekt=x["schaden_objekt"],
        ).with_inputs("anonymized_text")
        for x in so_test.to_dict("records")
    ]

    logger.info("Evaluating the dspy way")
    evaluate_program = Evaluate(
        devset=testset,
        metric=answer_exact_match,
        display_progress=True,
        display_table=10,
    )

    eval = evaluate_program(cot_objekt_predictor)
    print(eval)

    logger.success("Program finished successfully")
