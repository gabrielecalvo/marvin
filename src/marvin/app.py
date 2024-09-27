import logging
import warnings

from ._core import Marvin, load_models

logging.basicConfig(level=logging.INFO, format="✨MARVIN✨: %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Loading models..")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        marvin = Marvin(models=load_models())

    logger.info("Models loaded")

    while True:
        marvin.wait_for_wake_word()
        transcription = marvin.transcribe()
        response = marvin.query_llm(text=transcription)
        audio = marvin.synthesise(response)
        marvin.play_audio(audio)


if __name__ == "__main__":
    main()
