import torch, numpy as np
import asyncio
import threading
import time
import queue
import logging
from typing import Generator, Tuple
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from transformers import AutoTokenizer
import os

# Import SNAC for proper high-quality decoding
from snac import SNAC

# Setup logging
logger = logging.getLogger(__name__)

class VllmOrpheusModel:
    def __init__(self, model_name: str = "canopylabs/orpheus-3b-0.1-ft", dtype=torch.bfloat16, tokenizer='canopylabs/orpheus-3b-0.1-pretrained', **engine_kwargs):
        # Store configuration
        self.model_name = self._map_model_params(model_name)
        self.dtype = dtype
        self.sample_rate = 24000
        self.available_voices = ["zoe", "zac", "jess", "leo", "mia", "julia", "leah", "tara"]
        
        # Engine configuration
        self.engine_kwargs = {
            'max_model_len': engine_kwargs.get('max_model_len', 114000),
            'gpu_memory_utilization': engine_kwargs.get('gpu_memory_utilization', 0.9),
            'trust_remote_code': engine_kwargs.get('trust_remote_code', True),
            **{k: v for k, v in engine_kwargs.items() if k not in ['engine_use_ray', 'disable_log_requests']}
        }
        
        # Load tokenizer
        tokenizer_path = tokenizer if tokenizer else model_name
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        
        # Initialize SNAC decoder for production-quality audio
        self._setup_snac_decoder()
        
        # Setup single event loop for ALL requests
        self._loop = None
        self._loop_thread = None
        self._engine = None
        self._initialization_complete = threading.Event()
        self._shutdown_requested = threading.Event()
        
        # Initialize async infrastructure
        self._setup_single_event_loop()

    def _setup_snac_decoder(self):
        """Initialize SNAC decoder for production-quality audio output"""
        try:
            self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            self.snac_device = os.environ.get("SNAC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
            self.snac_model = self.snac_model.to(self.snac_device)
        except Exception as e:
            raise RuntimeError("SNAC decoder initialization failed - required for high-quality audio")

    def _turn_token_into_id(self, token_string, index):
        """Convert custom token string to SNAC token ID"""
        token_string = token_string.strip()
        
        # Find the last token in the string
        last_token_start = token_string.rfind("<custom_token_")
        
        if last_token_start == -1:
            return None
        
        # Extract the last token
        last_token = token_string[last_token_start:]
        
        # Process the last token
        if last_token.startswith("<custom_token_") and last_token.endswith(">"):
            try:
                number_str = last_token[14:-1]
                return int(number_str) - 10 - ((index % 7) * 4096)
            except ValueError:
                return None
        else:
            return None

    def _convert_to_audio(self, multiframe, count):
        """Convert SNAC tokens to high-quality audio with post-processing"""
        if len(multiframe) < 7:
            return None
        
        try:
            codes_0 = torch.tensor([], device=self.snac_device, dtype=torch.int32)
            codes_1 = torch.tensor([], device=self.snac_device, dtype=torch.int32)
            codes_2 = torch.tensor([], device=self.snac_device, dtype=torch.int32)

            num_frames = len(multiframe) // 7
            frame = multiframe[:num_frames*7]

            # Group tokens into SNAC's 3-layer hierarchy
            for j in range(num_frames):
                i = 7*j
                if codes_0.shape[0] == 0:
                    codes_0 = torch.tensor([frame[i]], device=self.snac_device, dtype=torch.int32)
                else:
                    codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=self.snac_device, dtype=torch.int32)])

                if codes_1.shape[0] == 0:
                    codes_1 = torch.tensor([frame[i+1]], device=self.snac_device, dtype=torch.int32)
                    codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=self.snac_device, dtype=torch.int32)])
                else:
                    codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=self.snac_device, dtype=torch.int32)])
                    codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=self.snac_device, dtype=torch.int32)])
                
                if codes_2.shape[0] == 0:
                    codes_2 = torch.tensor([frame[i+2]], device=self.snac_device, dtype=torch.int32)
                    codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=self.snac_device, dtype=torch.int32)])
                    codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=self.snac_device, dtype=torch.int32)])
                    codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=self.snac_device, dtype=torch.int32)])
                else:
                    codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=self.snac_device, dtype=torch.int32)])
                    codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=self.snac_device, dtype=torch.int32)])
                    codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=self.snac_device, dtype=torch.int32)])
                    codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=self.snac_device, dtype=torch.int32)])

            codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
            
            # Validate token ranges for quality assurance
            if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or 
                torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or 
                torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
                logger.warning("[SNAC] Invalid token range detected, skipping chunk")
                return None

            # Decode through SNAC
            with torch.inference_mode():
                audio_hat = self.snac_model.decode(codes)
            
            # Extract audio slice
            audio_slice = audio_hat[:, :, 2048:4096]
            detached_audio = audio_slice.detach().cpu()
            audio_np = detached_audio.numpy()
            
            # Production-quality post-processing
            # audio_np = self._apply_audio_post_processing(audio_np)
            
            # Convert to 16-bit PCM
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            return audio_bytes
            
        except Exception as e:
            logger.error(f"[SNAC] Audio conversion failed: {e}")
            return None

    # def _apply_audio_post_processing(self, audio_np):
    #     """Apply production-quality audio post-processing"""
    #     # 1. Normalization to prevent clipping
    #     max_val = np.max(np.abs(audio_np))
    #     if max_val > 0.95:
    #         audio_np = audio_np * (0.95 / max_val)
        
    #     # 2. Apply short fade in/out to prevent clicks/pops (5ms)
    #     fade_samples = int(0.005 * self.sample_rate)  # 5ms fade
    #     if audio_np.shape[-1] > fade_samples * 2:
    #         # Fade in
    #         fade_in = np.linspace(0, 1, fade_samples)
    #         audio_np[0, 0, :fade_samples] *= fade_in
    #         # Fade out
    #         fade_out = np.linspace(1, 0, fade_samples)
    #         audio_np[0, 0, -fade_samples:] *= fade_out
        
    #     # 3. Check for NaN/Inf values
    #     if np.isnan(audio_np).any() or np.isinf(audio_np).any():
    #         logger.warning("[PostProcess] NaN/Inf detected, applying safe fallback")
    #         audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.95, neginf=-0.95)
        
    #     return audio_np

    def _tokens_decoder_streaming(self, token_gen):
        """High-quality streaming SNAC decoder with sliding window"""
        buffer = []
        count = 0
        
        for token_sim in token_gen:       
            token = self._turn_token_into_id(token_sim, count)
            if token is None:
                continue
            
            if token > 0:
                buffer.append(token)
                count += 1

                # Decode when we have enough tokens (sliding window approach)
                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:]  # Last 4 frames (28 tokens)
                    audio_samples = self._convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples

    def _map_model_params(self, model_name):
        """Map model names to actual repo IDs"""
        model_map = {
            "medium-3b": {
                "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            },
        }
        unsupported_models = ["nano-150m", "micro-400m", "small-1b"]
        
        if model_name in unsupported_models:
            raise ValueError(f"Model {model_name} is not supported. Only medium-3b is supported")
        elif model_name in model_map:
            return model_map[model_name]["repo_id"]
        else:
            return model_name

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer with fallback handling"""
        try:
            if os.path.isdir(tokenizer_path):
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            return AutoTokenizer.from_pretrained("gpt2")

    def _setup_single_event_loop(self):
        """Setup single event loop for ALL requests - fixes concurrency issues"""
        def run_event_loop():
            try:
                # Create THE event loop for all operations
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                
                # Create vLLM engine in this loop
                async def create_engine():
                    engine_args = AsyncEngineArgs(
                        model=self.model_name,
                        dtype=self.dtype,
                        **self.engine_kwargs
                    )
                    self._engine = AsyncLLMEngine.from_engine_args(engine_args)
                    self._initialization_complete.set()
                
                # Initialize and run forever
                self._loop.run_until_complete(create_engine())
                
                # Keep loop running for all requests
                while not self._shutdown_requested.is_set():
                    try:
                        self._loop.run_until_complete(asyncio.sleep(0.1))
                    except:
                        break
                        
            except Exception as e:
                self._initialization_complete.set()
                raise
        
        # Start the single event loop thread
        self._loop_thread = threading.Thread(target=run_event_loop, daemon=True)
        self._loop_thread.start()
        
        # Wait for initialization
        if not self._initialization_complete.wait(timeout=300):
            raise RuntimeError("Engine initialization timed out")
            
        if self._engine is None:
            raise RuntimeError("Engine initialization failed")

    def _format_prompt(self, prompt, voice="leo", model_type="larger"):  # Default to leo for better quality
        """Format prompt exactly like orpheus_tts"""
        if model_type == "smaller":
            if voice:
                return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>"
            else:
                return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string
            else:
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string

    def generate_tokens_sync(self, prompt, voice=None, request_id="req-001", temperature=0.4, top_p=0.9, max_tokens=1200, stop_token_ids=[49158], repetition_penalty=1.1):
        """TRUE STREAMING version - yields tokens as they come from vLLM (optimized params for quality)"""
        prompt_string = self._format_prompt(prompt, voice)
        
        sampling_params = SamplingParams(
            temperature=temperature,    # Lower for stability
            top_p=top_p,               # Higher for diversity
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,  # Lower to reduce artifacts
        )

        # Create a queue for streaming tokens (like original orpheus_tts)
        token_queue = queue.Queue()

        async def async_producer():
            """Producer that feeds tokens into queue as they come"""
            try:
                async for result in self._engine.generate(
                    prompt=prompt_string, 
                    sampling_params=sampling_params, 
                    request_id=request_id
                ):
                    # Put token in queue immediately (streaming!)
                    token_queue.put(result.outputs[0].text)
                # Signal completion
                token_queue.put(None)  
            except Exception as e:
                print(f"[OrpheusModel] Token generation error for {request_id}: {e}")
                token_queue.put(None)  # Signal completion even on error
                raise

        # Submit async producer to shared event loop
        asyncio.run_coroutine_threadsafe(async_producer(), self._loop)
        
        # Yield tokens as they arrive in queue (TRUE STREAMING)
        while True:
            try:
                token = token_queue.get(timeout=30)  # Small timeout per token
                if token is None:  # Completion signal
                    break
                yield token
            except queue.Empty:
                print(f"[OrpheusModel] Token timeout for {request_id}")
                break
            except Exception as e:
                print(f"[OrpheusModel] Error getting token for {request_id}: {e}")
                break

    def generate_speech(self, **kwargs):
        """API-compatible speech generation using HIGH-QUALITY SNAC decoder"""
        return self._tokens_decoder_streaming(self.generate_tokens_sync(**kwargs))

    def generate_speech_streaming(self, text: str, voice: str = "leo", request_id: str = "default") -> Generator[Tuple[bytes, int, bool], None, None]:
        """Streaming interface with OPTIMIZED TTFB and HIGH-QUALITY SNAC decoding"""
        
        start_time = time.time()
        chunk_count = 0
        total_audio_bytes = 0
        
        try:
            # Now this will stream tokens as they come with high-quality SNAC decoding!
            audio_generator = self.generate_speech(
                prompt=text,
                voice=voice,
                request_id=request_id
            )
            
            for audio_chunk in audio_generator:
                chunk_count += 1
                
                # Convert to bytes
                if isinstance(audio_chunk, bytes):
                    chunk_bytes = audio_chunk
                elif isinstance(audio_chunk, np.ndarray):
                    chunk_bytes = audio_chunk.astype(np.int16).tobytes() if audio_chunk.dtype != np.int16 else audio_chunk.tobytes()
                else:
                    chunk_bytes = bytes(audio_chunk)
                
                if len(chunk_bytes) > 0:
                    total_audio_bytes += len(chunk_bytes)
                    
                    # Log TTFB for first chunk
                    if chunk_count == 1:
                        ttfb = (time.time() - start_time) * 1000
                    yield chunk_bytes, self.sample_rate, False
                
            # Signal completion with metrics
            duration = time.time() - start_time
            audio_duration = total_audio_bytes / (self.sample_rate * 2)  # 16-bit = 2 bytes per sample
            rtf = audio_duration / duration if duration > 0 else 0
            yield b'', self.sample_rate, True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield b'', self.sample_rate, True

    def shutdown(self):
        """Graceful shutdown"""
        self._shutdown_requested.set()
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=10)

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.shutdown()
        except:
            pass
