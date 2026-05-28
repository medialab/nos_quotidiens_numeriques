#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Programme :
- Appuyer sur ESPACE pour démarrer l'enregistrement audio.
- Réappuyer sur ESPACE pour arrêter l'enregistrement.
- Chaque enregistrement est stocké dans un dossier séparé dans recordings/.
- Le fichier audio est sauvegardé en .wav.
- L'audio est transcrit automatiquement avec faster-whisper.
- La transcription brute est sauvegardée en .txt.
- Un LLM local via Ollama produit un récit descriptif à partir de la transcription.
- Le récit généré est sauvegardé en .txt.
- Le récit, ou la transcription brute selon la configuration, est imprimé
  sur une imprimante thermique ESC/POS.

Dépendances :
pip install sounddevice soundfile numpy pynput faster-whisper python-escpos requests

Prérequis Ollama :
ollama pull mistral-small
"""

from __future__ import annotations

import queue
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
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

    # LLM local via Ollama
    enable_llm_rewriting: bool = True

    # Modèle Ollama.
    # Il faut l'avoir installé avec :
    # ollama pull mistral-small
    ollama_model: str = "mistral-small"

    # URL locale de l'API Ollama.
    ollama_url: str = "http://localhost:11434/api/chat"

    # Limite de génération du récit.
    llm_max_tokens: int = 900

    # Température faible pour éviter une réécriture trop imaginative.
    llm_temperature: float = 0.30

    # Temps maximal d'attente pour la réponse Ollama, en secondes.
    # Mistral-small peut être lent sur Mac M1.
    ollama_timeout: int = 300

    # Impression
    enable_printing: bool = True

    # Que faut-il imprimer ?
    # Valeurs possibles :
    # - "raw_transcription" : transcription brute
    # - "descriptive_narrative" : récit produit par le LLM
    print_mode: str = "descriptive_narrative"

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

        # Dossier racine : recordings/
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Dossier spécifique à cet enregistrement
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_dir = self.config.output_dir / f"recording_{timestamp}"
        recording_dir.mkdir(parents=True, exist_ok=False)

        # Fichier audio dans ce dossier
        audio_path = recording_dir / "audio.wav"

        sf.write(
            file=str(audio_path),
            data=audio_data,
            samplerate=self.config.samplerate,
        )

        print(f"Dossier d'enregistrement créé : {recording_dir}")
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
            print(
                f"Langue détectée : {info.language} "
                f"avec probabilité {info.language_probability:.2f}"
            )

        return transcription, txt_path


# ---------------------------------------------------------------------
# Génération locale d'un récit descriptif via Ollama
# ---------------------------------------------------------------------

class DescriptiveNarrativeGenerator:
    def __init__(self, config: AppConfig):
        self.config = config

        if not self.config.enable_llm_rewriting:
            return

        self.check_ollama_available()

    def check_ollama_available(self) -> None:
        """
        Vérifie qu'Ollama répond localement.
        """
        base_url = self.config.ollama_url.replace("/api/chat", "")

        try:
            response = requests.get(
                f"{base_url}/api/tags",
                timeout=5,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Impossible de joindre Ollama localement.\n"
                "Vérifiez que l'application Ollama est lancée, puis testez :\n\n"
                "curl http://localhost:11434/api/tags\n\n"
                f"Erreur technique : {exc}"
            ) from exc

    def build_messages(self, transcription: str) -> list[dict[str, str]]:
        system_prompt = (
            "Tu es un assistant d'écriture pour une enquête qualitative en sciences sociales. "
            "À partir d'une transcription orale brute, tu produis un récit descriptif, situé, "
            "sobre et attentif aux détails ordinaires, comme pourrait le faire un anthropologue "
            "ou un sociologue dans des notes de terrain rédigées. "
            "Tu ne dois pas inventer d'informations absentes de la transcription. "
            "Tu peux reformuler, organiser et rendre lisible, mais tu dois conserver les hésitations, "
            "incertitudes, affects, gestes mentionnés, situations pratiques, relations et problèmes "
            "tels qu'ils apparaissent dans le matériau. "
            "Ne psychologise pas excessivement. "
            "Ne conclus pas à la place de la personne. "
            "N'emploie pas de jargon inutile. "
            "Écris en français."
        )

        user_prompt = (
            "Voici une transcription brute issue d'un enregistrement audio :\n\n"
            "----- TRANSCRIPTION -----\n"
            f"{transcription}\n"
            "-------------------------\n\n"
            "Produis maintenant un récit descriptif en 2 à 6 paragraphes. "
            "Le texte doit être rédigé à la troisième personne si la situation s'y prête, "
            "ou sous une forme descriptive neutre si l'énonciateur n'est pas identifiable. "
            "Conserve une grande prudence interprétative. "
            "Ne mentionne pas que tu es un modèle de langage. "
            "Ne commence pas par une formule du type « Voici un récit ». "
            "Écris directement le récit."
        )

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    def generate(self, transcription: str, source_txt_path: Path) -> tuple[str, Path]:
        if not self.config.enable_llm_rewriting:
            print("Réécriture par LLM désactivée.")
            return transcription, source_txt_path

        narrative_path = source_txt_path.with_name(
            source_txt_path.stem + "_recit.txt"
        )

        if not transcription.strip():
            print("Transcription vide : aucun récit généré.")
            narrative_path.write_text("", encoding="utf-8")
            return "", narrative_path

        print(
            "Génération du récit descriptif avec Ollama "
            f"et le modèle : {self.config.ollama_model}"
        )

        payload = {
            "model": self.config.ollama_model,
            "messages": self.build_messages(transcription),
            "stream": False,
            "options": {
                "temperature": self.config.llm_temperature,
                "num_predict": self.config.llm_max_tokens,
            },
        }

        try:
            response = requests.post(
                self.config.ollama_url,
                json=payload,
                timeout=self.config.ollama_timeout,
            )
            response.raise_for_status()

        except requests.RequestException as exc:
            raise RuntimeError(
                "Erreur lors de l'appel à Ollama.\n"
                "Vérifiez que le modèle est bien téléchargé avec :\n\n"
                f"ollama pull {self.config.ollama_model}\n\n"
                f"Erreur technique : {exc}"
            ) from exc

        data = response.json()

        try:
            narrative = data["message"]["content"].strip()
        except KeyError as exc:
            raise RuntimeError(
                "Réponse Ollama inattendue. "
                f"Réponse reçue : {data}"
            ) from exc

        narrative_path.write_text(narrative, encoding="utf-8")

        print(f"Récit descriptif sauvegardé : {narrative_path}")

        return narrative, narrative_path


# ---------------------------------------------------------------------
# Impression thermique
# ---------------------------------------------------------------------

class ThermalPrinter:
    def __init__(self, config: AppConfig):
        self.config = config

    def print_text(self, text: str, title: str = "RECIT DESCRIPTIF") -> None:
        if not self.config.enable_printing:
            print("Impression désactivée dans la configuration.")
            return

        if not text.strip():
            print("Texte vide : rien à imprimer.")
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

            printer.text(f"{title}\n")
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
    #     printer.text("RECIT DESCRIPTIF\n")
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
        self.narrative_generator = DescriptiveNarrativeGenerator(config)
        self.printer = ThermalPrinter(config)
        self.is_processing = False
        self.should_quit = False

    def process_recording(self, audio_path: Path) -> None:
        self.is_processing = True

        try:
            transcription, txt_path = self.transcriber.transcribe(audio_path)

            print("\n--- Transcription brute ---")
            print(transcription)
            print("---------------------------\n")

            narrative, narrative_path = self.narrative_generator.generate(
                transcription=transcription,
                source_txt_path=txt_path,
            )

            print("\n--- Récit descriptif ---")
            print(narrative)
            print("------------------------\n")

            if self.config.print_mode == "raw_transcription":
                text_to_print = transcription
                title = "TRANSCRIPTION"
            elif self.config.print_mode == "descriptive_narrative":
                text_to_print = narrative
                title = "RECIT DESCRIPTIF"
            else:
                raise ValueError(
                    "print_mode doit valoir "
                    "'raw_transcription' ou 'descriptive_narrative'."
                )

            self.printer.print_text(text_to_print, title=title)

        except Exception as exc:
            print("Erreur pendant le traitement de l'enregistrement.", file=sys.stderr)
            print(str(exc), file=sys.stderr)

        finally:
            self.is_processing = False
            print(
                "Prêt. Appuyez sur ESPACE pour enregistrer à nouveau, "
                "ou ÉCHAP pour quitter."
            )

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
                print(
                    "Traitement en cours. Veuillez attendre la fin "
                    "avant de relancer un enregistrement."
                )
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
        print("Réappuyez sur ESPACE pour arrêter, transcrire, réécrire et imprimer.")
        print("Appuyez sur ÉCHAP pour quitter.")

        with keyboard.Listener(on_press=self.on_key_press) as listener:
            listener.join()


# ---------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------

def main() -> None:
    config = AppConfig(
        # Dossier de sortie
        output_dir=Path("data"),

        # Modèle de transcription speech-to-text
        whisper_model_size="base",
        whisper_device="cpu",
        whisper_compute_type="int8",
        language="fr",

        # LLM local via Ollama
        enable_llm_rewriting=True,
        ollama_model="mistral",
        ollama_url="http://localhost:11434/api/chat",
        llm_max_tokens=900,
        llm_temperature=0.30,
        ollama_timeout=300,

        # Impression
        enable_printing=False,

        # Choix du texte imprimé :
        # "raw_transcription" ou "descriptive_narrative"
        print_mode="descriptive_narrative",

        # À modifier selon votre imprimante
        printer_vendor_id=0x04b8,
        printer_product_id=0x0202,
    )

    app = VoiceToThermalPrinterApp(config)
    app.run()


if __name__ == "__main__":
    main()