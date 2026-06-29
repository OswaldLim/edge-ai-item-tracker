# Edge AI Item Tracker

Wearable egocentric memory system that tracks personal items using on-device vision. Fully local, privacy-first.

## How It Works

1. Camera (webcam / phone / Jetson + cam) captures frames
2. YOLOv8-nano detects target items (keys, wallet, glasses)
3. Detections stored in ChromaDB with timestamp, CLIP embedding, location label
4. User queries via Telegram: *"Where's my wallet?"* → gets location + thumbnail

## Architecture

```
Camera → YOLO detection → Embedding generation → ChromaDB
                                                          ↓
User (Telegram) → RAG query → LLM response + thumbnail
```

## Phase Roadmap

| Phase | Goal | Cost |
|-------|------|------|
| 1 | Single item, single location, Telegram bot | $0 |
| 2 | Multi-item tracking | $0 |
| 3 | Wearable/egocentric capture | ~$30 |
| 4 | Multi-room with BLE beacons | ~$45 |\ | VLM scene understanding (Jetson) | ~$150 |
| 6 | Proactive alerts + pattern learning | ~$20 |
| 7 | Custom wearable hardware | ~$50 |

## Tech Stack

- **Detection**: YOLOv8-nano → ONNX / TensorRT
- **Embeddings**: CLIP (openai/clip-ViT-B-32)
- **Vector DB**: ChromaDB
- **RAG**: Local LLM (llama.cpp / LiteRT-LM)
- **Bot**: python-telegram-bot
- **Dataset**: Roboflow

## Getting Started

### Phase 1 Setup
```bash
pip install ultralytics opencv-python chromadb clip-by-openai python-telegram-bot
python src/detect.py --source 0  # webcam
```

## Hardware

See Notion Hardware DB for live tracking.

## Credits

Inspired by: Meta Aria Project, Rewind Pin, Edge Impulse TinyML projects