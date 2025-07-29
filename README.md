<div align="center">

  <a href="https://tensorfuse.io/"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./assets/light.png">
    <img alt="stts logo" src="./assets/light.png" height="110" style="max-width: 100%;">
  </picture></a>
  
<a href="https://join.slack.com/t/tensorfusecommunity/shared_invite/zt-30r6ik3dz-Rf7nS76vWKOu6DoKh5Cs5w"><img src="./assets/join_slack.png" width="165"></a>
<a href="https://tensorfuse.io/docs/concepts/introduction"><img src="./assets/Documentation Button.png" width="137"></a>

### STTS is the world's first open-source audio inference server built for voice models. It lets you self-host the latest open-source STT/TTS models with streaming API in your own infrastructure. 


</div>

## ✨ Supported Models

List of supported models along with the TTFB for each. If you want us to add more models, email at founders@tensorfuse.io

| Model Name | Type | TTFB | GPU Used |
|-----------|---------|--------|----------|
| **Orpheus(3B)**     | TTS | 180 ms | 1xH100 |
| **Whisper** (coming soon)     | STT | ----  | ---- |

## ⚡ Quickstart

- **Run with Docker (recommended)** for Linux devices:
The easiest way to run the server is by running the following docker command in your terminal. It will start the streaming server on port 7004.

```bash
docker run --gpus=all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8003:8003 -p 7004:7004 \
  --shm-size=1g \
  -e HUGGING_FACE_HUB_TOKEN=hf_XXXX \
  -v "${HOME}/.cache/huggingface":/cache/huggingface \
  tensorfuse/stts:latest --mode tts --model orpheus
```

You can start streaming audio from the server using the script below

```python
import requests
import sseclient  # Make sure this is from sseclient-py (pip install sseclient-py)
import logging
import base64
import wave
import json
import os
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("sse-debug")
API_URL = "http://localhost:7004/v1/audio"
TEXT = "<giggle> The quick brown fox jumps over the lazy dog"
MODEL = "orpheus"
OUT = "debug_audio.wav"

def debug_sse():
    params = {"text": TEXT, "model": MODEL}
    logger.info(f"Requesting: {API_URL} with {params}")
    try:
        response = requests.post(API_URL, params=params, stream=True)
        logger.info(f"HTTP {response.status_code}")
        logger.info(f"Headers: {dict(response.headers)}")
        logger.info(f"Content-Type: {response.headers.get('content-type','')}")
        if 'text/event-stream' not in response.headers.get('content-type', ''):
            logger.warning("Server did NOT send the expected text/event-stream Content-Type!")

        client = sseclient.SSEClient(response)  # Do not read from response first!
        print("Connected to SSE, waiting for events:")
        all_bytes = bytearray()
        sample_rate = None

        for event in client.events():
            logger.info(f"Received SSE event: {event.data[:70]}{'...' if len(event.data) > 70 else ''}")
            if event.data == '[DONE]':
                logger.info("Stream finished, got [DONE].")
                break
            try:
                data = json.loads(event.data)
                if 'audio' in data:
                    chunk = base64.b64decode(data['audio'])
                    all_bytes.extend(chunk)
                    if not sample_rate:
                        sample_rate = data.get('rate')
            except Exception as e:
                logger.error(f"Failed to parse SSE data: {event.data} ({e})")

        if all_bytes and sample_rate:
            with wave.open(OUT, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sample_rate)
                f.writeframes(all_bytes)
            logger.info(f"Saved audio to {os.path.abspath(OUT)}")
        else:
            logger.warning("No audio chunks or sample rate detected.")

    except Exception as e:
        logger.exception(f"Error during SSE client run: {e}")

if __name__ == "__main__":
    debug_sse()
```


## About

This repo is built and maintained by Tensorfuse. We're building serverless GPU runtime that lets you self-host AI models of any modality in your own AWS. 

If you're building AI agents, Voice Agents, Chatbots, etc and want to customize and deploy models, check out [Tensorfuse](https://tensorfuse.io)

Below is the list of all resource to help you get started


## 🔗 Links and Resources
| Type                            | Links                               |
| ------------------------------- | --------------------------------------- |
| **Get Started**              | [Start here](https://app.tensorfuse.io/) |
| 📚 **Documentation**              | [Read Our Docs](https://tensorfuse.io/docs/concepts/introduction) |
| <img width="16" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg" />&nbsp; **Twitter (aka X)**              |  [Follow us on X](https://x.com/tensorfuse)|
| 🔮 **Model Library**            | [SOTA models hosted on AWS using Tensorfuse](https://tensorfuse.io/docs/guides/modality/text/llama_4)|
| ✍️ **Blog**                    | [Read our Blogs](https://tensorfuse.io/docs/blogs/blog)|


For any enquries, email at founders@tensorfuse.io



