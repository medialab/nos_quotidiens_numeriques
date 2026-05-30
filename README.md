
# Installation

Install PortAudio:

```
# macOS
brew install portaudio

# Debian / Ubuntu
sudo apt install portaudio19-dev
```

Setup venv:

```
python3 -m venv .venv
source .venv/bin/activate
```

Install ollama and mistral :

```sh
ollama pull mistral
```

Install dependencies:

```sh

pip install sounddevice soundfile numpy pynput faster-whisper python-escpos requests
```

Locate printer: 

```bash
lsusb
```

Then modifiy code:

```python
printer_vendor_id=0x04b8
printer_product_id=0x0202
```

(todo make a piped bash command to lsub then pass on the ids)