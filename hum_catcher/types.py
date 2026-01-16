import dataclasses
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Any
import numpy as np

@dataclasses.dataclass
class WaveData:
    """An ENF signal with its associated attributes."""
    nominal_hz: float
    real_hz: float
    samples: np.ma.masked_array 
    start_time: Optional[datetime] = None

    @property
    def step_time(self) -> timedelta:
        return timedelta(seconds=1.0 / self.real_hz)

    @property
    def total_time(self) -> timedelta:
        return self.step_time * len(self.samples)

    @property
    def end_time(self) -> Optional[datetime]:
        if self.start_time is not None:
            return self.start_time + self.total_time
        return None

    def get_quality(self) -> float:
        return 1.0 - np.mean(self.samples.mask)

@dataclasses.dataclass
class SyncAttempt:
    """A single result of a time matching comparison between two ENF signals."""
    time_shift: timedelta
    span: timedelta
    similarity: float
    error_val: float
    final_score: float

@dataclasses.dataclass
class ExtractedInfo:
    """The result of the detection of an ENF signal from an audio file."""
    hum_wave: WaveData
    freq_map: Tuple[np.ndarray, np.ndarray, np.ndarray] 
    noise_ratio: np.ndarray
    base_harmonic: int
    other_harmonics: List[int] = dataclasses.field(default_factory=list)