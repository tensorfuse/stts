# models/orpheus/1/model.py
import threading, numpy as np
import triton_python_backend_utils as pb_utils
from mytts.models.vllm_orpheus import VllmOrpheusModel

class TritonPythonModel:
    def initialize(self, args):
        # Load Orpheus once at startup
        self.model = VllmOrpheusModel()
        # Ensure decoupled policy is enabled
        # config = pb_utils.get_model_config()
        # if not pb_utils.using_decoupled_model_transaction_policy(config):
        #     raise pb_utils.TritonModelException("Decoupled mode not enabled")

    def execute(self, requests):
        for request in requests:
            request_id = request.request_id()
            # Extract input text
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            text = text_tensor.as_numpy()[0].decode('utf-8')
            sender = request.get_response_sender()
            # Stream in background thread
            threading.Thread(target=self._stream_audio, args=(sender, text, request_id), daemon=True).start()
        return None  # responses sent asynchronously

    def _stream_audio(self, sender, text, request_id):
        for chunk, rate, final in self.model.generate_speech_streaming(text = text, request_id = request_id):
            audio_int16 = np.frombuffer(chunk, dtype=np.int16)       # raw PCM bytes
            t_audio = pb_utils.Tensor("audio", audio_int16)          # Triton sees TYPE_INT16
            # audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            # t_audio = pb_utils.Tensor("audio", audio_np)
            t_rate  = pb_utils.Tensor("sampling_rate", np.array([rate], dtype=np.int32))
            t_last = pb_utils.Tensor("is_last", np.array([bool(final)], dtype=np.bool_))
            response = pb_utils.InferenceResponse(output_tensors=[t_audio, t_rate, t_last])
            if final:
                # Send final response with completion flag
                sender.send(response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                break
            else:
                sender.send(response, flags=0)