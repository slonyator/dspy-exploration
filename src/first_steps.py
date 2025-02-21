import os

from dotenv import load_dotenv
from dspy import Signature, InputField, OutputField, Predict, settings, LM
from typing import Literal, Annotated
from loguru import logger
from pydantic import BeforeValidator


def konfidenz_validator(value):
    if not 0 <= value <= 1:
        raise ValueError("Konfidenz muss zwischen 0 und 1 liegen")


class InformationExtractor(Signature):
    """Informations-Extraktion aus Schadenmeldungen"""

    text: str = InputField(desc="Text der Schadenmeldung")
    melder: Literal[
        "Versicherungsnehmer", "Aussendienst", "Anspruchsteller", "Sonstige"
    ] = OutputField(desc="Person / Institution, die den Schaden gemeldet hat")
    zusamenfassung: str = OutputField(
        desc="Zusammenfassung der Schadenmeldung"
    )
    schadendatum: str = OutputField(desc="Datum des Schadens")
    konfidenz: Annotated[float, BeforeValidator(konfidenz_validator)] = (
        OutputField(desc="Konfidenz der Vorhersage")
    )


if __name__ == "__main__":
    _ = load_dotenv()

    lm = LM(
        model="openai/gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        max_tokens=1000,
    )
    settings.configure(lm=lm)

    logger.info("Model settings finished")
    sample_text = "Der Versicherungsnehmer meldet einen Schaden am 01.01.2023."

    information = Predict(InformationExtractor)
    result = information(text=sample_text).toDict()

    logger.info("Prediction finished")

    print(result)
