# **Qwen3-TTS-Daggr-UI**

> A Gradio-based demonstration for the Qwen/Qwen3-TTS-12Hz models using Daggr for modular UI nodes. Supports voice design (prompt-to-speech), voice cloning (zero-shot), and custom voice synthesis with multiple speakers and languages. Features lazy model loading to optimize memory, multi-model sizes (0.6B and 1.7B), and support for various audio inputs.

https://github.com/user-attachments/assets/8bc46f5d-e09c-4077-91c4-a1970697cb64
https://github.com/user-attachments/assets/51e0236d-b0d6-47f0-9829-99fcccde4d1c
https://github.com/user-attachments/assets/2246d4c7-3f5c-49b8-a177-4bd5305eca48

## Features
- **Modular UI with Daggr**: Three independent nodes for Voice Design, Voice Clone, and Custom Voice, allowing parallel or sequential workflows.
- **Lazy Model Loading**: Models (Base, CustomVoice, VoiceDesign) load on-demand based on task and size, minimizing VRAM usage.
- **Advanced TTS Tasks**:
  - Voice Design: Generate speech from text and voice description prompts.
  - Voice Clone: Zero-shot cloning from reference audio (with optional transcript).
  - Custom Voice: Synthesis using predefined speakers with optional style instructions.
- **Multi-Language Support**: Auto-detection or manual selection for Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian.
- **Model Sizes**: 0.6B for faster inference, 1.7B for higher quality.
- **Audio Handling**: Supports file paths, base64 data URIs, numpy arrays, and normalization to [-1, 1] float32.
- **Efficient Inference**: Uses bfloat16 on CUDA, with garbage collection and cache clearing for low-memory setups.
- **Speakers**: 9 built-in options including Aiden, Dylan, Eric, Ono_anna, Ryan, Serena, Sohee, Uncle_fu, Vivian.
- **Queueing**: Up to default Gradio queue limits for concurrent jobs.

## Prerequisites
- Python 3.10 or higher.
- CUDA-compatible GPU (recommended for bfloat16 and faster inference).
- Stable internet for initial model downloads from Hugging Face.
- Hugging Face token (HF_TOKEN) set as an environment variable for gated models.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/Qwen3-TTS-Daggr-UI.git
   cd Qwen3-TTS-Daggr-UI
   ```
2. Install dependencies:
   First, install pre-requirements:
   ```
   pip install -r pre-requirements.txt
   ```
   Then, install main requirements:
   ```
   pip install -r requirements.txt
   ```
   **pre-requirements.txt content:**
   ```
   pip>=23.0.0
   ```
   **requirements.txt content:**
   ```
   daggr==0.5.2
   torch==2.8.0
   transformers==4.57.3
   accelerate==1.12.0
   einops
   gradio #gradio@6
   librosa
   torchaudio
   soundfile
   sox
   onnxruntime
   spaces
   torch
   numpy
   ```
3. Set Hugging Face token (if required):
   ```
   export HF_TOKEN=your_hf_token_here
   ```
4. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860`.

### Docker Installation
1. Build the Docker image:
   ```
   docker build -t qwen3-tts-daggr-ui .
   ```
2. Run the container:
   ```
   docker run -p 7860:7860 --env HF_TOKEN=your_hf_token_here qwen3-tts-daggr-ui
   ```
   Access at `http://localhost:7860`.

## Usage
1. **Select a Node**: Use the Daggr UI to interact with Voice Design, Voice Clone, or Custom Voice sections.
2. **Input Parameters**:
   - For Voice Design: Provide text, language, and voice description.
   - For Voice Clone: Upload reference audio, optional transcript, target text, language, and model size.
   - For Custom Voice: Provide text, language, speaker, optional style instruction, and model size.
3. **Generate Audio**: Click the submit button in the respective node to produce output.
4. **Output**: Audio waveform (numpy) and status message.

### Supported Speakers
| Speaker    | Description |
|------------|-------------|
| Aiden     | General-purpose |
| Dylan     | General-purpose |
| Eric      | General-purpose |
| Ono_anna  | Japanese-style |
| Ryan      | General-purpose |
| Serena    | General-purpose |
| Sohee     | Korean-style |
| Uncle_fu  | General-purpose |
| Vivian    | General-purpose |

### Supported Languages
Auto, Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian.

## Examples
| Task          | Input Example | Parameters |
|---------------|---------------|------------|
| Voice Design | "It's in the top drawer... wait, it's empty? No way!" | Language: Auto, Description: "Speak in an incredulous tone, but with a hint of panic." |
| Custom Voice | "Hello! Welcome to the Text-to-Speech demo." | Language: English, Speaker: Ryan, Instruction: "Neutral" , Model: 1.7B |
| Voice Clone  | Target: "Cloned voice speaking this." | Ref Audio: Upload WAV/MP3, Ref Text: "Reference speech.", Language: Auto, x-vector only: False, Model: 1.7B |

## Troubleshooting
- **Model Loading**: First use downloads models; monitor console for progress.
- **OOM Errors**: Switch to smaller model (0.6B), reduce max_new_tokens, or clear cache manually.
- **Audio Input Issues**: Ensure uploads are valid WAV/MP3; check console for processing errors.
- **No Output**: Provide required inputs (e.g., text, audio); ensure HF_TOKEN is set for gated repos.
- **CUDA Fallback**: If no GPU, runs on CPU with float32 (slower).

## Contributing
Contributions welcome! Add new nodes to the Daggr graph, support more models/speakers, or improve audio processing. Submit pull requests via the repository.

Repository: [https://github.com/PRITHIVSAKTHIUR/Qwen3-TTS-Daggr-UI.git](https://github.com/PRITHIVSAKTHIUR/Qwen3-TTS-Daggr-UI.git)

## License
Apache License 2.0. See [LICENSE](LICENSE) for details.
Built by Prithiv Sakthi. Report issues via the repository.
