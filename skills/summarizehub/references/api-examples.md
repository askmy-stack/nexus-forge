# SummarizeHub API Examples

Base URL: `http://localhost:8080`

Start the server:

```bash
cd nexus-forge
uv sync --extra multimodal
uv run uvicorn textSummarizer.serving.app:app --reload --port 8080
```

## Health

```bash
curl http://localhost:8080/health
```

```json
{"status": "ok", "version": "1.0.0", "models_available": 7}
```

## List models

```bash
curl http://localhost:8080/models
```

## Text summarization

```bash
curl -X POST http://localhost:8080/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence is reshaping healthcare. Machine learning detects disease from scans.",
    "model": "extractive",
    "strategy": "map_reduce",
    "max_length": 128
  }'
```

## Multimodal (JSON)

### Text via multimodal endpoint

```bash
curl -X POST http://localhost:8080/summarize/multimodal \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "text",
    "text": "Long article text here...",
    "model": "extractive",
    "strategy": "stuff",
    "max_length": 128
  }'
```

### Image (base64)

```bash
IMG_B64=$(base64 -i photo.png)
curl -X POST http://localhost:8080/summarize/multimodal \
  -H "Content-Type: application/json" \
  -d "{
    \"input_type\": \"image\",
    \"base64_data\": \"${IMG_B64}\",
    \"model\": \"extractive\",
    \"max_length\": 128
  }"
```

### Audio (server-side path)

```bash
curl -X POST http://localhost:8080/summarize/multimodal \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "audio",
    "path": "/absolute/path/to/recording.wav",
    "model": "extractive",
    "max_length": 128
  }'
```

## Multimodal (file upload)

```bash
curl -X POST http://localhost:8080/summarize/multimodal/upload \
  -F "input_type=image" \
  -F "model=extractive" \
  -F "strategy=stuff" \
  -F "max_length=128" \
  -F "file=@photo.png"
```

```bash
curl -X POST http://localhost:8080/summarize/multimodal/upload \
  -F "input_type=audio" \
  -F "model=extractive" \
  -F "file=@recording.wav"
```

### Video (server-side path)

```bash
curl -X POST http://localhost:8080/summarize/multimodal \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "video",
    "path": "/absolute/path/to/demo.mp4",
    "model": "extractive",
    "strategy": "map_reduce",
    "max_length": 128
  }'
```

```bash
curl -X POST http://localhost:8080/summarize/multimodal/upload \
  -F "input_type=video" \
  -F "model=extractive" \
  -F "strategy=map_reduce" \
  -F "max_length=128" \
  -F "file=@demo.mp4;type=video/mp4"
```

## Grade summary

```bash
curl -X POST http://localhost:8080/grade \
  -H "Content-Type: application/json" \
  -d '{
    "source": "AI transforms healthcare and finance. Machine learning powers diagnostics.",
    "summary": "AI transforms healthcare and finance.",
    "threshold": 3.5
  }'
```

```json
{
  "score": {
    "coherence": 4,
    "faithfulness": 4,
    "fluency": 4,
    "relevance": 3,
    "average": 3.75,
    "feedback": "Summary meets baseline quality."
  },
  "passes": true,
  "threshold": 3.5
}
```

## Error responses

Unknown model (422):

```json
{"detail": "Unknown model 'gpt-4'. Available: bart, extractive, flan-t5, longt5, pegasus, pegasus-xsum, t5"}
```

Invalid strategy (422):

```json
{"detail": [{"type": "string_pattern_mismatch", "loc": ["body", "strategy"], "msg": "..."}]}
```
