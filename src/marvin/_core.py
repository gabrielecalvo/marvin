import logging
import sys
import warnings
from collections import namedtuple
from collections.abc import Callable
from typing import Any

import sounddevice
import torch
from datasets import load_dataset
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor, pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

PLAYBACK_SAMPLE_RATE = 17_000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)
Models = namedtuple("Models", ["classifier", "transcriber", "synthesiser", "replier"])


def _generate_synthesiser_function() -> Callable[[str], torch.Tensor]:
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    synth_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(DEVICE)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    def synthesiser(text: str) -> torch.Tensor:
        inputs = processor(text=text, return_tensors="pt")
        speech = synth_model.generate_speech(
            inputs["input_ids"].to(DEVICE),
            speaker_embeddings.to(DEVICE),
            vocoder=vocoder,
        )
        return speech.cpu()

    return synthesiser


def load_models(
    classifier_model: str | None = "MIT/ast-finetuned-speech-commands-v2",
    transcriber_model: str | None = "openai/whisper-small.en",
    replier_model: str | None = "Qwen/Qwen2.5-3B-Instruct",
) -> Models:
    classifier = transcriber = replier = None
    if classifier_model:
        classifier = pipeline("audio-classification", model=classifier_model, device=DEVICE)
    if transcriber_model:
        transcriber = pipeline("automatic-speech-recognition", model=transcriber_model, device=DEVICE)
    if replier_model:
        replier = pipeline("text-generation", model=replier_model, device=DEVICE)

    # for now the synthesiser is not going to be configurable
    synthesiser = _generate_synthesiser_function()

    return Models(classifier=classifier, transcriber=transcriber, synthesiser=synthesiser, replier=replier)


class Marvin:
    models: Models

    def __init__(self, models: Models):
        self.models = models

    @classmethod
    def from_model_names(cls, **kw: Any) -> "Marvin":
        return cls(load_models(**kw))

    def wait_for_wake_word(
        self,
        wake_word: str = "marvin",
        prob_threshold: float = 0.5,
        chunk_length_s: float = 2.0,
        stream_chunk_s: float = 0.25,
    ) -> bool:
        logger.info("Wake system setup...")

        classifier = self.models.classifier
        if wake_word not in classifier.model.config.label2id:
            raise ValueError(
                f"Wake word {wake_word} not in set of valid class labels, "
                f"pick a wake word in the set {classifier.model.config.label2id.keys()}."
            )

        sampling_rate = classifier.feature_extractor.sampling_rate

        mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=chunk_length_s,
            stream_chunk_s=stream_chunk_s,
        )

        logger.info("Listening for wake word...")
        for prediction in classifier(mic):
            _value = prediction[0]
            logger.debug(_value)
            if _value["label"] == wake_word and _value["score"] > prob_threshold:
                logger.info("Wake word detected!")
                return True

        return False

    def transcribe(
        self,
        chunk_length_s: float = 5.0,
        stream_chunk_s: float = 1.0,
    ) -> str:
        transcriber = self.models.transcriber
        sampling_rate = transcriber.feature_extractor.sampling_rate

        mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=chunk_length_s,
            stream_chunk_s=stream_chunk_s,
        )

        logger.info(f"Starting {chunk_length_s}s transcription, you can now speak 🎤...")
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            for item in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
                sys.stdout.write("\033[K")
                print(item["text"], end="\r")
                if not item["partial"][0]:
                    break

        logger.info(f"Transcription ended {item['text']}")
        return item["text"]

    def synthesise(self, text: str) -> torch.Tensor:
        logger.info(f"Synthesising text: {text}")
        return self.models.synthesiser(text)

    def query_llm(self, text: str) -> str:
        logger.info(f"Querying local LLM with: {text}")
        chat_prompts = [
            {
                "role": "system",
                "content": "You are a helpful assistant who replies "
                "to user requests in a very succinct and targeted way.",
            },
            {"role": "user", "content": text},
        ]
        llm_response = self.models.replier(chat_prompts, max_new_tokens=50)
        return llm_response[0]["generated_text"][-1]["content"]

    @staticmethod
    def play_audio(audio_array: torch.Tensor) -> None:
        logger.info(f"Playing audio array of length {audio_array.shape[0]}..")
        sounddevice.play(audio_array, samplerate=PLAYBACK_SAMPLE_RATE, blocking=True)
        logger.info("Audio playback complete")
