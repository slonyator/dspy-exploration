from dotenv import load_dotenv
from dspy import Signature, InputField, OutputField, Predict
from typing import Literal


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

    sample_text = "Der Versicherungsnehmer meldet einen Schaden am 01.01.2023."

    information = Predict(InformationExtractor)
    result = information(text=sample_text)

    print(result)
