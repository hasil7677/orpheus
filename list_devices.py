"""
list_devices.py — Print available audio devices and their index numbers.

Usage:
    .venv\\Scripts\\python.exe list_devices.py

Copy the index number of the mic/speaker you want into local/.env as
AUDIO_INPUT_DEVICE / AUDIO_OUTPUT_DEVICE (see .env.example).
"""

import pyaudio

p = pyaudio.PyAudio()

default_in = p.get_default_input_device_info()["index"]
default_out = p.get_default_output_device_info()["index"]

print(f"{'Idx':<5}{'In':<4}{'Out':<4}{'Default':<9}Name")
print("-" * 70)
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    is_default = []
    if i == default_in:
        is_default.append("in")
    if i == default_out:
        is_default.append("out")
    print(
        f"{i:<5}{info['maxInputChannels']:<4}{info['maxOutputChannels']:<4}"
        f"{','.join(is_default):<9}{info['name']}"
    )

p.terminate()
