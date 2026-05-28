#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Programme :
- Appuyer sur ESPACE pour démarrer l'enregistrement audio.
- Réappuyer sur ESPACE pour arrêter l'enregistrement.
- Le fichier audio est sauvegardé en .wav.
- L'audio est transcrit automatiquement avec faster-whisper.
- La transcription est sauvegardée en .txt.
- La transcription est imprimée sur une imprimante thermique ESC/POS.

Dépendances :
pip install sounddevice soundfile numpy pynput faster-whisper python-escpos
"""

from __future__ import annotations

import queue
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from pynput import keyboard

# Import ESC/POS
# Pour une imprimante USB :
from escpos.printer import Usb

# Alternative possible si l'imprimante est configurée dans CUPS :
# from escpos.printer import CupsPrinter


# ---------------------------------------------------------------------
# Configuration générale
# ---------------------------------------------------------------------

@dataclass
class AppConfig:
    output_dir: Path = Path("recordings")

    # Audio
    samplerate: int = 16000
    channels: int = 1
    dtype: str = "float32"

    # Touches
    record_key: keyboard.Key = keyboard.Key.space
    quit_key: keyboard.Key = keyboard.Key.esc

    # Whisper / faster-whisper
    # Options possibles : "tiny", "base", "small", "medium", "large-v3"
    # "base" ou "small" sont de bons compromis sur CPU.
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    language: str = "fr"

    # Impression
    enable_printing: bool = True

    # Paramètres USB ESC/POS.
    # À modifier selon votre imprimante.
    printer_vendor_id: int = 0x04b8
    printer_product_id: int = 0x0202

    # Si nécessaire, ajuster ces paramètres.
    printer_timeout: int = 0
    printer_in_ep: int = 0x82
    printer_out_ep: int = 0x01


# ---------------------------------------------------------------------
# Enregistreur audio
# ---------------------------------------------------------------------

class AudioRecorder:
    def __init__(self, config: AppConfig):
        self.config = config
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.frames: list[np.ndarray] = []
        self.stream: Optional[sd.InputStream] = None
        self.is_recording: bool = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[Audio warning] {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def start(self) -> None:
        if self.is_recording:
            print("Un enregistrement est déjà en cours.")
            return

        self.frames = []
        self.audio_queue = queue.Queue()

        self.stream = sd.InputStream(
            samplerate=self.config.samplerate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            callback=self._callback,
        )

        self.stream.start()
        self.is_recording = True
        print("Enregistrement démarré. Appuyez sur ESPACE pour arrêter.")

    def stop_and_save(self) -> Path:
        if not self.is_recording:
            raise RuntimeError("Aucun enregistrement en cours.")

        assert self.stream is not None

        self.stream.stop()
        self.stream.close()
        self.stream = None
        self.is_recording = False

        while not self.audio_queue.empty():
            self.frames.append(self.audio_queue.get())

        if not self.frames:
            raise RuntimeError("Aucune donnée audio enregistrée.")

        audio_data = np.concatenate(self.frames, axis=0)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = self.config.output_dir / f"recording_{timestamp}.wav"

        sf.write(
            file=str(audio_path),
            data=audio_data,
            samplerate=self.config.samplerate,
        )

        print(f"Fichier audio sauvegardé : {audio_path}")
        return audio_path


# ---------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------

class Transcriber:
    def __init__(self, config: AppConfig):
        self.config = config
        print(f"Chargement du modèle Whisper : {config.whisper_model_size}")
        self.model = WhisperModel(
            config.whisper_model_size,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )

    def transcribe(self, audio_path: Path) -> tuple[str, Path]:
        print("Transcription en cours...")

        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.config.language,
            beam_size=5,
            vad_filter=True,
        )

        lines: list[str] = []

        for segment in segments:
            text = segment.text.strip()
            if text:
                lines.append(text)

        transcription = "\n".join(lines).strip()

        txt_path = audio_path.with_suffix(".txt")
        txt_path.write_text(transcription, encoding="utf-8")

        print(f"Transcription sauvegardée : {txt_path}")

        if info.language:
            print(f"Langue détectée : {info.language} avec probabilité {info.language_probability:.2f}")

        return transcription, txt_path


# ---------------------------------------------------------------------
# Impression thermique
# ---------------------------------------------------------------------

class ThermalPrinter:
    def __init__(self, config: AppConfig):
        self.config = config

    def print_text(self, text: str) -> None:
        if not self.config.enable_printing:
            print("Impression désactivée dans la configuration.")
            return

        if not text.strip():
            print("Transcription vide : rien à imprimer.")
            return

        print("Connexion à l'imprimante thermique...")

        try:
            printer = Usb(
                idVendor=self.config.printer_vendor_id,
                idProduct=self.config.printer_product_id,
                timeout=self.config.printer_timeout,
                in_ep=self.config.printer_in_ep,
                out_ep=self.config.printer_out_ep,
            )

            printer.set(
                align="left",
                font="a",
                width=1,
                height=1,
                density=9,
            )

            printer.text("TRANSCRIPTION\n")
            printer.text("--------------------------------\n")
            printer.text(text)
            printer.text("\n\n")
            printer.cut()

            print("Impression terminée.")

        except Exception as exc:
            print("Erreur lors de l'impression.", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            print(
                "\nVérifiez :\n"
                "- que l'imprimante est branchée ;\n"
                "- que vendor_id et product_id sont corrects ;\n"
                "- que l'utilisateur a les droits USB nécessaires ;\n"
                "- que l'imprimante est compatible ESC/POS.",
                file=sys.stderr,
            )

    # Variante possible si l'imprimante est configurée via CUPS :
    #
    # def print_text_cups(self, text: str, printer_name: str) -> None:
    #     printer = CupsPrinter(printer_name)
    #     printer.text("TRANSCRIPTION\n")
    #     printer.text("--------------------------------\n")
    #     printer.text(text)
    #     printer.text("\n\n")
    #     printer.cut()


# ---------------------------------------------------------------------
# Application principale
# ---------------------------------------------------------------------

class VoiceToThermalPrinterApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.recorder = AudioRecorder(config)
        self.transcriber = Transcriber(config)
        self.printer = ThermalPrinter(config)
        self.is_processing = False
        self.should_quit = False

    def process_recording(self, audio_path: Path) -> None:
        self.is_processing = True

        try:
            transcription, txt_path = self.transcriber.transcribe(audio_path)

            print("\n--- Transcription ---")
            print(transcription)
            print("---------------------\n")

            self.printer.print_text(transcription)

        except Exception as exc:
            print("Erreur pendant le traitement de l'enregistrement.", file=sys.stderr)
            print(str(exc), file=sys.stderr)

        finally:
            self.is_processing = False
            print("Prêt. Appuyez sur ESPACE pour enregistrer à nouveau, ou ÉCHAP pour quitter.")

    def on_key_press(self, key) -> Optional[bool]:
        if key == self.config.quit_key:
            print("Arrêt demandé.")
            self.should_quit = True

            if self.recorder.is_recording:
                try:
                    self.recorder.stop_and_save()
                except Exception:
                    pass

            return False

        if key == self.config.record_key:
            if self.is_processing:
                print("Traitement en cours. Veuillez attendre la fin avant de relancer un enregistrement.")
                return None

            if not self.recorder.is_recording:
                self.recorder.start()
            else:
                try:
                    audio_path = self.recorder.stop_and_save()
                    self.process_recording(audio_path)
                except Exception as exc:
                    print("Erreur lors de l'arrêt de l'enregistrement.", file=sys.stderr)
                    print(str(exc), file=sys.stderr)

        return None

    def run(self) -> None:
        print("Programme lancé.")
        print("Appuyez sur ESPACE pour démarrer l'enregistrement.")
        print("Réappuyez sur ESPACE pour arrêter, transcrire et imprimer.")
        print("Appuyez sur ÉCHAP pour quitter.")

        with keyboard.Listener(on_press=self.on_key_press) as listener:
            listener.join()


# ---------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------

def main() -> None:
    config = AppConfig(
        # Dossier de sortie
        output_dir=Path("recordings"),

        # Modèle de transcription
        whisper_model_size="large-v3",
        whisper_device="cpu",
        whisper_compute_type="int8",
        language="fr",

        # Impression
        enable_printing=False,

        # À modifier selon votre imprimante
        printer_vendor_id=0x04b8,
        printer_product_id=0x0202,
    )

    app = VoiceToThermalPrinterApp(config)
    app.run()


if __name__ == "__main__":
    main()