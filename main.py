import pyaudio
import wave
import os
import whisper
from threading import Thread
from queue import Queue

model = whisper.load_model("medium")
CHUNK_SIZE = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "temp.wav"

class Scribe:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.running = True
        self.audio_queue = Queue()
        self.stream = None

    def capture_audio(self):
        while self.running:
            frames = []
            for _ in range(0, int(RATE / CHUNK_SIZE * RECORD_SECONDS)):
                if not self.running:
                    break
                try:
                    data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)
                except IOError as e:
                    print(f"Error capturing audio: {e}")
            if frames:
                self.audio_queue.put(frames)

    def process_audio(self):
        while self.running:
            frames = self.audio_queue.get()
            if frames is None:
                break
            try:
                with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(self.p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                result = model.transcribe(WAVE_OUTPUT_FILENAME, fp16=False)
                print(">>>\n", result["text"], "\n")
            except Exception as e:
                print(f"Error processing audio: {e}")
            finally:
                if os.path.exists(WAVE_OUTPUT_FILENAME):
                    os.remove(WAVE_OUTPUT_FILENAME)

    def main(self):
        try:
            self.stream = self.p.open(
                format=FORMAT, 
                channels=CHANNELS, 
                rate=RATE, 
                input=True, 
                frames_per_buffer=CHUNK_SIZE
            )

            capture_thread = Thread(target=self.capture_audio)
            process_thread = Thread(target=self.process_audio)
            
            capture_thread.start()
            process_thread.start()

            print("... recording in progress ... enter 'ok' to stop recording ...")
            while self.running:
                if input().lower() == 'ok':
                    self.running = False

            self.audio_queue.put(None)
            capture_thread.join()
            process_thread.join()

        except Exception as e:
            print(f"Error running main loop: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.p.terminate()

if __name__ == "__main__":
    Scribe().main()
