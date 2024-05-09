import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def inference(audio):
    # Hyperparameters
    chunk_length = 2
    print(f"Chunk length: {chunk_length} seconds")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=chunk_length,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(audio, generate_kwargs={"task":"transcribe"})
    for segment in result["chunks"]:
        print(segment)