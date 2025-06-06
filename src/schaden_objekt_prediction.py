import os
from typing import Literal

import dspy
import pandas as pd
from dotenv import load_dotenv
from dspy.evaluate import Evaluate
from dspy.teleprompt.bootstrap import BootstrapFewShot
from loguru import logger
from pyprojroot import here
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


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


def result_exact_match(example, prediction, trace=None):
    logger.info(
        f"example: {example}, prediction: {prediction}, trace: {trace}"
    )
    return example.result == prediction.result


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

    logger.info("Mapping predictions to abbreviations for object")
    objekt_predictions = [
        Mapper.get_object_abbreviation(pred.result)
        for pred in objekt_predictions
    ]

    logger.info("Calculating accuracy")
    objekt_accuracy = accuracy_score(
        test["schaden_objekt"], [pred for pred in objekt_predictions]
    )
    logger.info(f"Schaden-Objekt accuracy: {objekt_accuracy}")

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

    logger.info(lm.inspect_history(n=1))

    logger.info("Starting with FS Prompt optimization for Schaden-Objekt")
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
            text=x["anonymized_text"], result=x["schaden_objekt"]
        ).with_inputs("text")
        for x in so_train.to_dict("records")
    ]
    testset = [
        dspy.Example(
            text=x["anonymized_text"], result=x["schaden_objekt"]
        ).with_inputs("text")
        for x in so_test.to_dict("records")
    ]

    logger.info("Evaluating the dspy way")

    evaluate_program = Evaluate(
        devset=testset,
        metric=result_exact_match,
        display_progress=True,
        display_table=10,
    )

    eval = evaluate_program(cot_objekt_predictor)
    logger.info(f"Evaluation results: {eval}")

    logger.info("Setting up BootstrapFewShot teleprompter")
    teleprompter = BootstrapFewShot(
        metric=result_exact_match, max_labeled_demos=10
    )

    logger.info("Compiling predictor with few-shot learning")
    compiled_predictor = teleprompter.compile(
        cot_objekt_predictor, trainset=trainset
    )

    logger.info("Evaluating compiled predictor on test set")
    evaluate_program = Evaluate(
        devset=testset,
        metric=result_exact_match,
        num_threads=8,
        display_progress=True,
        display_table=10,
    )

    logger.info("Run the evaluation after compilation")
    eval_compiled = evaluate_program(compiled_predictor)
    logger.info(f"Evaluation Result: {eval_compiled}")

    logger.info(lm.inspect_history(n=1))

    logger.success("Program finished successfully")
