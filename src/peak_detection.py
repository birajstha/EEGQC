from time import sleep

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from mne_lsl.stream import StreamLSL

ECG_HEIGHT: float = 98.0  # percentile height constraint, in %
ECG_DISTANCE: float = 0.5  # distance constraint, in seconds


class Detector:
    """Real-time single channel peak detector.

    Parameters
    ----------
    bufsize : float
        Size of the buffer in seconds. The buffer will be filled on instantiation, thus
        the program will hold during this duration.
    stream_name : str
        Name of the LSL stream to use for the respiration or cardiac detection. The
        stream should contain a respiration channel using a respiration belt or a
        thermistor and/or an ECG channel.
    stream_source_id : str | None
        A unique identifier of the device or source of the data. If not empty, this
        information improves the system robustness since it allows recipients to recover
        from failure by finding a stream with the same source_id on the network.
    ch_name : str
        Name of the ECG channel in the LSL stream. This channel should contain the ECG
        signal recorded with 2 bipolar electrodes.
    """

    def __init__(
        self,
        bufsize: float,
        stream_name: str,
        stream_source_id: str | None,
        ch_name: str,
    ) -> None:
        # create stream
        self._stream = StreamLSL(
            bufsize, name=stream_name, source_id=stream_source_id
        ).connect(acquisition_delay=None, processing_flags="all")
        self._stream.pick(ch_name)
        self._stream.set_channel_types({ch_name: "misc"}, on_unit_change="ignore")
        self._stream.notch_filter(50, picks=ch_name)
        self._stream.notch_filter(100, picks=ch_name)
        sleep(bufsize)  # prefill an entire buffer
        # peak detection settings
        self._last_peak = None
        self._peak_candidates = None
        self._peak_candidates_count = None

    def detect_peaks(self) -> NDArray[np.float64]:
        """Detect all peaks in the buffer.

        Returns
        -------
        peaks : array of shape (n_peaks,)
            The timestamps of all detected peaks.
        """
        self._stream.acquire()
        if self._stream.n_new_samples == 0:
            return np.array([])  # nothing new to do
        data, ts = self._stream.get_data()  # we have a single channel in the stream
        data = data.squeeze()
        peaks, _ = find_peaks(
            data,
            distance=ECG_DISTANCE * self._stream.info["sfreq"],
            height=np.percentile(data, ECG_HEIGHT),
        )
        return ts[peaks]

    def new_peak(self) -> float | None:
        """Detect new peak entering the buffer.

        Returns
        -------
        peak : float | None
            The timestamp of the newly detected peak. None if no new peak is detected.
        """
        ts_peaks = self.detect_peaks()
        if ts_peaks.size == 0:
            return None
        if self._peak_candidates is None and self._peak_candidates_count is None:
            self._peak_candidates = list(ts_peaks)
            self._peak_candidates_count = [1] * ts_peaks.size
            return None
        peaks2append = []
        for k, peak in enumerate(self._peak_candidates):
            if peak in ts_peaks:
                self._peak_candidates_count[k] += 1
            else:
                peaks2append.append(peak)
        # before going further, let's make sure we don't add too many false positives,
        # which could be indicative of noise in the signal (e.g. movements)
        if int(self._stream._bufsize * (1 / ECG_DISTANCE)) < len(peaks2append) + len(
            self._peak_candidates
        ):
            self._peak_candidates = None
            self._peak_candidates_count = None
            return None
        self._peak_candidates.extend(peaks2append)
        self._peak_candidates_count.extend([1] * len(peaks2append))
        # now, all the detected peaks have been triage, let's see if we have a winner
        idx = [k for k, count in enumerate(self._peak_candidates_count) if 4 <= count]
        if len(idx) == 0:
            return None
        peaks = sorted([self._peak_candidates[k] for k in idx])
        # compare the winner with the last known peak
        if self._last_peak is None:  # don't return the first peak detected
            new_peak = None
            self._last_peak = peaks[-1]
        if self._last_peak is None or self._last_peak + ECG_DISTANCE <= peaks[-1]:
            new_peak = peaks[-1]
            self._last_peak = peaks[-1]
        else:
            new_peak = None
        # reset the peak candidates
        self._peak_candidates = None
        self._peak_candidates_count = None
        return new_peak

    @property
    def stream(self):
        """Stream object."""
        return self._stream


from mne_lsl.lsl import local_clock
import uuid
from matplotlib import pyplot as plt

name = "ecg-example"
source_id = "c0ffee1234567890deadbeefcafebabe"
detector = Detector(4, name, source_id, "AUX8")
delays = list()
while len(delays) <= 30:
    peak = detector.new_peak()
    if peak is not None:
        delays.append((local_clock() - peak) * 1e3)
detector.stream.disconnect()

f, ax = plt.subplots(1, 1, layout="constrained")
ax.set_title("Detection delay in ms")
ax.hist(delays, bins=15)
plt.show()