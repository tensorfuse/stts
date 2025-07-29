import os, base64, numpy as np, asyncio
from functools import partial
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import tritonclient.grpc as grpcclient
import json
import logging
import sys
import uuid
import io, wave

# Configure logging at the top of your file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add this to test logging works
logger.info("Server starting up...")

app = FastAPI()

@app.get("/readiness")
async def readiness():
    return {"status": "ready"}

@app.post("/v1/audio")
async def tts(text: str, model: str = "orpheus"):
    if model not in ("orpheus", "sesame"):
        raise HTTPException(400, "Unsupported model")
    
    # Create a unique request ID for each request
    request_id = f"req_{uuid.uuid4().hex[:8]}"

    client = None

    client = grpcclient.InferenceServerClient(
        url=os.getenv("TRITON_URL", "localhost:8001"),
        verbose=False
    )
    
    text_bytes = text.encode("utf-8")
    # Prepare input tensor
    inp = grpcclient.InferInput("text", [1], "BYTES")
    inp.set_data_from_numpy(np.array([text_bytes], dtype=np.object_))

    # 1) Get the current running loop (main event loop)
    loop = asyncio.get_running_loop()  # safe inside coroutine  [oai_citation:2‡docs.python.org](https://docs.python.org/3/library/asyncio-eventloop.html?utm_source=chatgpt.com)

    # 2) Create an asyncio.Queue for thread-safe handoff
    resp_q: asyncio.Queue = asyncio.Queue()

    # 3) Define callback that schedules queue.put on the main loop
    def _callback(result, error):
        # This runs in Triton's gRPC thread—use call_soon_threadsafe
        if error:
            logger.error(f"[TTS] Callback received error: {error.message()}")
        loop.call_soon_threadsafe(resp_q.put_nowait, (result, error))

    # 5) Async generator to yield SSE chunks
    async def _generator():
        stream_started = False
        try:
            # 4) Start the gRPC stream with our callback
            client.start_stream(callback=_callback)
            stream_started = True
            client.async_stream_infer(
                model_name=model,
                inputs=[inp],
                outputs=[
                    grpcclient.InferRequestedOutput("audio"),
                    grpcclient.InferRequestedOutput("sampling_rate"),
                    grpcclient.InferRequestedOutput("is_last")
                    ],
                request_id=request_id
            )
            while True:
                result, error = await resp_q.get()
                if error:
                    logger.error(f"[TTS] Error received: {error.message()}")
                    payload = {
                        "error":   error.message(),
                        "status":  error.status(),
                        "details": error.details()
                        }
                    yield f"data: {json.dumps(payload)}\n\n"
                    break
                try:
                    # Check if this is a completion-only response (no data)
                    if result is None:
                        yield "data: [DONE]\n\n"
                        break
                    audio = result.as_numpy("audio").tobytes()
                    rate  = int(result.as_numpy("sampling_rate")[0])
                    is_last = bool(result.as_numpy("is_last")[0])
                    if len(audio) > 0:
                        chunk = base64.b64encode(audio).decode()
                        yield f"data: {json.dumps({'audio': chunk, 'rate': rate})}\n\n"

                    if is_last:
                        yield "data: [DONE]\n\n"
                        break
                except Exception as e:
                    logger.error(f"[TTS] Error processing result: {e}", exc_info=True)
                    yield f"data: {{\"error\": \"Processing failed: {str(e)}\"}}\n\n"
                    break
        except Exception as e:
            logger.error(f"[TTS] Generator exception: {e}", exc_info=True)
            yield f"data: {{\"error\": \"Generator failed: {str(e)}\"}}\n\n"
        finally:
            if stream_started:
                try:
                    client.stop_stream()
                except Exception as e:
                    logger.error(f"[TTS] Error stopping stream: {e}")
            try:
                client.close()
            except Exception as e:
                logger.error(f"[TTS] Error closing client: {e}")


    return StreamingResponse(_generator(), media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache"})