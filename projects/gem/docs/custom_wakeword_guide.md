# Custom Wake Word Guide

Gem's VoiceAgent is now configured to dynamically load custom wake word models! You can name your AI absolutely anything you want (like "Ooga Booga", "Brog", or "Computer").

Here is exactly how to generate your own custom model in about 2 minutes for free:

## 1. Generate the Model File

Training a wake word model from scratch normally requires a lot of code and synthetic audio generation. Luckily, the creators of openWakeWord provide a free, automated cloud generator!

1. Go to the [openWakeWord HuggingFace Space](https://huggingface.co/spaces/davidscripka/openWakeWord).
2. Type in your desired wake word phrase (e.g. `ooga_booga`). Note: it works best if you type it as you would say it phonetically.
3. Wait about 1-2 minutes for their cloud servers to generate the synthetic training data and build your model.
4. Download the resulting `.onnx` file (e.g., `ooga_booga.onnx`).

## 2. Install Your Model

1. Take your downloaded `.onnx` file and place it in the `Gem/Back/models/` directory. (If the directory doesn't exist, create it).
2. Open your `Gem/Back/.env` file.
3. Add or update the `WAKE_WORD` variable to match the exact base name of your `.onnx` file (without the `.onnx` extension).

For example, if your file is named `ooga_booga.onnx`, your `.env` should look like this:

```env
GEMINI_API_KEY=AIzaSy...
WAKE_WORD=ooga_booga
```

## 3. Restart Gem

Restart the terminal (`python main.py`). The VoiceAgent will automatically scan your `Back/models/` directory, see that the `WAKE_WORD` matches the `.onnx` file, and load it into memory!

You're done! Brog the caveman is now listening for your custom name.
