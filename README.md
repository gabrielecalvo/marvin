# Marvin

Marvin is a very basic AI assistant based on huggingface transformer models.

The project is mostly based on this [tutorial](https://huggingface.co/learn/audio-course/en/chapter7/voice-assistant).

It is able to run locally and:
- wake up on a keyword (defaulted to "marvin")
- transcribe speech to text
- answer questions using a transformer model
- convert the model's answer to audio
- play the audio

## Running locally
Assuming you installed it using `uv sync`, use `uv run marvin` to start the assistant.

After the initial model loading, you can wake it by saying "marvin".
After a moment, you will have 5 seconds to ask your question.

Use CTRL+C to stop the assistant.
