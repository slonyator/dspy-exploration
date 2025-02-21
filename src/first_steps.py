import os

from dotenv import load_dotenv
from dspy import Signature, InputField, OutputField, Predict, settings, LM
from typing import Literal
from loguru import logger


class InformationExtractor(Signature):
    text: str = InputField(desc="Text der Schadenmeldung")
    melder: Literal[
        "Versicherungsnehmer", "Aussendienst", "Anspruchsteller", "Sonstige"
    ] = OutputField(desc="Person / Institution, die den Schaden gemeldet hat")
    zusamenfassung: str = OutputField(
        desc="Zusammenfassung der Schadenmeldung"
    )
    schadendatum: str = OutputField(desc="Datum des Schadens")


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
    result = information(text=sample_text)

    logger.info("Prediction finished")

    print(result)
