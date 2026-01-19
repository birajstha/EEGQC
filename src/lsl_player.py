import uuid
from mne_lsl.datasets import sample
import multiprocessing as mp
import time

def player_process(fname, name, source_id, started_event, stop_event):
    """Process which runs the player and signals readiness via started_event."""
    from mne_lsl.player import PlayerLSL

    player = PlayerLSL(fname, chunk_size=1, name=name, source_id=source_id)
    player.start()
    started_event.set()
    try:
        # wait until parent asks us to stop
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        player.stop()


fname = sample.data_path() / "sample-ecg-raw.fif"
name = "ecg-example"
source_id = "c0ffee1234567890deadbeefcafebabe"

started = mp.Event()
stop = mp.Event()
process = mp.Process(target=player_process, args=(fname, name, source_id, started, stop))
process.start()

# wait for the player to actually start
started.wait()

try:
    print("Player started. Press Ctrl-C to stop.")
    while process.is_alive():
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    stop.set()
    process.join(timeout=5)