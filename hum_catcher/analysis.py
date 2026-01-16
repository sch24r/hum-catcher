import math
import enum
import numpy as np
import scipy.signal
import scipy.ndimage
import audiofile
import ffmpeg
import tempfile
import os
from datetime import timedelta
from typing import List, Tuple, Optional
from .types import ExtractedInfo, WaveData

FreqMap = Tuple[np.ndarray, np.ndarray, np.ndarray]

def grab_audio(loc: str) -> Tuple[np.ndarray, float]:
    try:
        raw, rate = audiofile.read(loc, always_2d=True)
        return np.mean(raw, axis=0), float(rate)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_name = tmp.name
        
        try:
            (
                ffmpeg
                .input(loc)
                .output(tmp_name, ac=1, vn=None, loglevel="quiet")
                .overwrite_output()
                .run()
            )
            raw, rate = audiofile.read(tmp_name, always_2d=True)
            return np.mean(raw, axis=0), float(rate)
        except ffmpeg.Error as err:
            raise RuntimeError(f"FFmpeg error: {err.stderr.decode() if err.stderr else str(err)}")
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

SMALLEST_CALC_HZ = 1000.0
SLICE_WIDTH = 0.25
BLOCK_TIME = timedelta(seconds=16)
NORM_TIME = timedelta(seconds=24)
TARGET_HUM_HZ = 1.0
SMOOTH_AMT = 3.5
GOOD_SIG_LEVEL = 2.85
GOOD_SIG_LEN = timedelta(seconds=8)
POOR_SIG_LEVEL = 1.6
SLOPE_MAX = 0.0025
BAD_BIT_LEN = timedelta(seconds=0)
QUAL_FLOOR = 0.075
SIM_FLOOR = 0.5
LEN_FLOOR = 0.05

def dig_for_hum(
    raw_samps: np.ndarray, samp_rate: float, grid_target: float = 50.0,
    harmonics_to_check: List[int] = [1, 2, 3, 4, 5, 6, 7, 8],
) -> Optional[ExtractedInfo]:
    
    len_secs = len(raw_samps) / samp_rate
    needed = BLOCK_TIME.total_seconds()
    
    if len_secs < needed:
        print(f"Error: Media file is too short ({len_secs:.2f}s < {needed}s).")
        return None

    small_samps, small_rate = _shrink_samps(raw_samps, samp_rate)

    maps = _build_freq_maps(
        small_samps, small_rate, grid_target, harmonics_to_check
    )

    outcomes = [
        _trace_hum_path(h_map, grid_target, h_num)
        for h_num, h_map in zip(harmonics_to_check, maps)
    ]

    good_ones = [
        (rez, f_map)
        for rez, f_map in zip(outcomes, maps)
        if rez.hum_wave.get_quality() >= QUAL_FLOOR
    ]

    if len(good_ones) == 0:
        return None
    else:
        return _mix_harmonics(grid_target, good_ones)

def _shrink_samps(samps: np.ndarray, rate: float) -> Tuple[np.ndarray, float]:
    if rate <= SMALLEST_CALC_HZ:
        return samps, rate

    factor = int(rate // SMALLEST_CALC_HZ)
    if factor <= 1:
        return samps, rate

    new_rate = rate / factor
    assert new_rate >= SMALLEST_CALC_HZ

    return scipy.signal.decimate(samps, factor, ftype="fir", n=16), new_rate

def _build_freq_maps(
    samps: np.ndarray, rate: float, base_hz: float,
    h_list: List[int],
) -> List[FreqMap]:
    ranges = [
        (
            h * (base_hz - SLICE_WIDTH),
            h * (base_hz + SLICE_WIDTH)
        )
        for h in h_list
    ]
    cleaned = _pass_filter(samps, rate, ranges, passes=10)
    return _do_fourier(cleaned, rate, ranges)

def _pass_filter(
    samps: np.ndarray, rate: float, cuts: List[Tuple[int, int]], passes: int
) -> np.ndarray:
    accum = None
    for lo, hi in cuts:
        half_rate = 0.5 * rate
        norm_lo = lo / half_rate
        norm_hi = hi / half_rate
        flt = scipy.signal.butter(passes, [norm_lo, norm_hi], analog=False, btype='band', output='sos')
        part = scipy.signal.sosfilt(flt, samps)
        if accum is None:
            accum = part
        else:
            accum += part
    return accum

def _do_fourier(
    samps: np.ndarray, rate: float, cuts: List[Tuple[int, int]],
) -> List[FreqMap]:
    CHUNK = 256
    tot_secs = len(samps) / rate
    win_secs = min(tot_secs, BLOCK_TIME.total_seconds())
    pts_per_seg = int(rate * win_secs)
    pts_overlap = int(pts_per_seg - rate / TARGET_HUM_HZ)

    t_list = []
    z_list = []
    hop = pts_per_seg - pts_overlap
    out_sz = math.ceil(len(samps) / hop) + 1
    half_win = pts_per_seg // 2

    for start in range(0, out_sz, CHUNK):
        end = start + CHUNK
        s_idx = max(
            0,
            int(start / TARGET_HUM_HZ * rate - half_win)
        )
        e_idx = int(end / TARGET_HUM_HZ * rate + half_win)
        bit = samps[s_idx:e_idx]

        if len(bit) < pts_per_seg:
            break

        if s_idx == 0:
            out_start = 0
        else:
            out_start = int(half_win / rate * TARGET_HUM_HZ)

        freqs, times, amps = scipy.signal.stft(bit, rate, nperseg=pts_per_seg, noverlap=pts_overlap, scaling="psd")
        amps = abs(amps)**2
        mask = None

        for lo, hi in cuts:
            sub = (freqs >= lo) & (freqs <= hi)
            if mask is None:
                mask = sub
            else:
                mask |= sub

        freqs = freqs[mask]
        t_list.append(times[out_start:out_start+CHUNK] + start / TARGET_HUM_HZ)
        z_list.append(amps[mask, out_start:out_start+CHUNK])

    final_t = np.concatenate(t_list)
    final_z = np.concatenate(z_list, axis=1)

    return [
        (freqs[(freqs >= lo) & (freqs <= hi)], final_t, final_z[(freqs >= lo) & (freqs <= hi)])
        for lo, hi in cuts
    ]

def _flatten_map(f_map: FreqMap, rate: float) -> FreqMap:
    ff, tt, zz = f_map
    if NORM_TIME is None:
        avg = np.mean(zz)
        dev = np.std(zz)
        return np.abs((zz - avg) / dev)

    w_size = round(NORM_TIME.total_seconds() * rate)
    zz = np.abs(zz).transpose()
    flat = np.empty(zz.shape)

    for i in range(0, len(zz)):
        w_start = max(0, i - w_size // 2)
        w_end = min(len(zz), i + w_size // 2)
        win = zz[w_start:w_end]
        avg = np.mean(win)
        dev = np.std(win)
        if dev == 0.0:
            flat[i] = 0.0
        else:
            flat[i] = (zz[i] - avg) / dev

    return ff, tt, np.abs(flat).transpose()

def _trace_hum_path(
    f_map: FreqMap, base_hz: float, h_num: int,
    extra_h: List[int] = [],
) -> ExtractedInfo:
    clean_map = _flatten_map(f_map, TARGET_HUM_HZ)
    ff, tt, zz = clean_map

    if len(ff) < 2 or len(tt) < 1:
        sp = timedelta(seconds=tt[-1] - tt[0])
        raise ValueError(f"unable to compute spectrum on signal of duration {sp}.")

    f_step = ff[1] - ff[0]
    trace = np.empty(len(tt), dtype=np.float16)
    ratios = np.empty(len(tt), dtype=np.float32)

    for i, col in enumerate(zz.transpose()):
        peak = np.amax(col)
        p_idx = np.where(col == peak)[0][0]
        real_peak = ff[0] + _quad_fit(col, p_idx, f_step)
        trace[i] = real_peak / h_num - base_hz
        avg = np.mean(col)
        if avg == 0.0:
            ratios[i] = 0
        else:
            ratios[i] = peak / avg

    trace = _clean_up_trace(trace, ratios)
    trace_obj = WaveData(
        nominal_hz=base_hz,
        samples=trace,
        real_hz=TARGET_HUM_HZ,
    )
    return ExtractedInfo(
        hum_wave=trace_obj,
        freq_map=clean_map,
        noise_ratio=ratios,
        base_harmonic=h_num,
        other_harmonics=extra_h,
    )

def _quad_fit(arr, idx, step):
    mid = arr[idx]
    l = arr[idx - 1] if idx > 0 else mid
    r = arr[idx + 1] if idx + 1 < len(arr) else mid
    bot = l - 2 * mid + r
    if bot == 0.0:
        return mid
    offset = 0.5 * (l - r) / bot
    return (idx + offset) * step

def _clean_up_trace(data: np.ndarray, snr: np.ndarray) -> np.ma.masked_array:
    soft = _smooth_trace(data)
    cut = np.ma.masked_where(
        (soft < -SLICE_WIDTH) | (soft > SLICE_WIDTH),
        soft
    )
    cut.fill_value = np.nan
    final = _cull_bad_spots(cut, snr)
    return final

def _smooth_trace(data: np.ma.masked_array) -> np.ma.masked_array:
    if SMOOTH_AMT is None:
        return data
    return scipy.ndimage.gaussian_filter1d(
        data.astype(np.float64), sigma=SMOOTH_AMT
    ).astype(np.float16)

def _cull_bad_spots(data: np.ma.masked_array, snr: np.ndarray) -> np.ma.masked_array:
    min_len = TARGET_HUM_HZ * GOOD_SIG_LEN.total_seconds()
    max_bad_len = TARGET_HUM_HZ * BAD_BIT_LEN.total_seconds()
    max_slope = SLOPE_MAX / TARGET_HUM_HZ

    def is_cut(k: int) -> bool:
        return np.ma.is_masked(data) and data.mask[k]
    def is_weak(k: int) -> bool:
        return snr[k] >= POOR_SIG_LEVEL
    def is_strong(k: int) -> bool:
        return snr[k] >= GOOD_SIG_LEVEL
    def is_long_enough(dur: int) -> bool:
        return dur >= min_len
    
    slope = np.abs(np.gradient(data))
    def slope_ok(k: int) -> bool:
        return slope[k] <= max_slope

    class State(enum.Enum):
        LOOKING = 1
        HAVE_STRONG = 2
        IN_GOOD = 3
        IN_BAD = 4

    mode = State.LOOKING
    run_len = None
    bad_len = None
    bad_slope_sum = None
    rejects = np.ma.masked_all(len(data)).mask
    i = 0
    while i < len(data):
        if mode == State.LOOKING:
            if not is_cut(i) and is_strong(i) and slope_ok(i):
                run_len = 1
                mode = State.HAVE_STRONG
        elif mode == State.HAVE_STRONG:
            if not is_cut(i) and is_strong(i) and slope_ok(i):
                run_len += 1
                if is_long_enough(run_len):
                    mode = State.IN_GOOD
                    rejects[i] = False
                    bad_len = 0
                    bad_slope_sum = 0
                    j = i - 1
                    while j >= 0 and rejects[j] and bad_len <= max_bad_len:
                        bad_len += 1
                        bad_slope_sum += slope[i]
                        if (not is_cut(j) and is_weak(j) and slope_ok(j) and
                            bad_slope_sum / bad_len < max_slope):
                            rejects[j] = False
                            bad_len = 0
                            bad_slope_sum = 0
                        j -= 1
            else:
                mode = State.LOOKING
        elif mode == State.IN_GOOD:
            if not is_cut(i) and is_weak(i) and slope_ok(i):
                rejects[i] = False
            else:
                mode = State.IN_BAD
                bad_len = 1
                bad_slope_sum = slope[i]
        elif mode == State.IN_BAD:
            if (not is_cut(i) and is_weak(i) and slope_ok(i) and
                (bad_slope_sum / bad_len < max_slope or is_strong(i))):
                rejects[i] = False
                mode = State.IN_GOOD
            else:
                bad_len += 1
                bad_slope_sum += slope[i]
                if bad_len > max_bad_len:
                    rejects[i] = False
                    mode = State.LOOKING
        i += 1
    return np.ma.array(data, mask=data.mask | rejects, fill_value=np.nan)

def _mix_harmonics(
    hz: float,
    items: List[Tuple[ExtractedInfo, FreqMap]]
) -> ExtractedInfo:
    from .match import _calc_one_sim
    if len(items) < 1:
        raise ValueError("at least one harmonic analysis result is required.")
    ranked = sorted(items, key=lambda r: r[0].base_harmonic)
    best_rez, best_map = ranked[0]
    xtra_h = []
    
    for rez, f_map in ranked[1:]:
        score, size = _calc_one_sim(best_rez.hum_wave.samples, rez.hum_wave.samples)
        frac = size / len(best_rez.hum_wave.samples)
        if score < SIM_FLOOR or frac < LEN_FLOOR:
            continue
        merged_map = _fuse_maps(
            f_map, rez.base_harmonic,
            best_map, best_rez.base_harmonic,
        )
        merged_xtra = xtra_h + [rez.base_harmonic]
        merged_rez = _trace_hum_path(
            merged_map, hz, best_rez.base_harmonic,
            merged_xtra,
        )
        if merged_rez.hum_wave.get_quality() > best_rez.hum_wave.get_quality():
            best_rez = merged_rez
            xtra_h = merged_xtra 
    return best_rez

def _fuse_maps(
    src: FreqMap, src_h: int,
    dst: FreqMap, dst_h: int,
) -> FreqMap:
    s_f, s_t, s_z = src
    d_f, d_t, d_z = dst
    s_f = s_f / src_h
    d_f = d_f / dst_h
    out_z = d_z.transpose()
    for t, _ in enumerate(s_t):
        out_z[t] = out_z[t] + np.interp(d_f, s_f, s_z[:,t])
    return dst[0], d_t, out_z.transpose()