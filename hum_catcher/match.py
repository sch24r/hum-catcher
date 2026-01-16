import enum
import math
import numpy as np
import time
from datetime import timedelta
from typing import List, Optional, Tuple, Callable
from tqdm import tqdm

from .types import WaveData, SyncAttempt
from . import ui

class CalcMethod(enum.Enum):
    NUMPY = "numpy"
    OPENCL = "opencl"

MATCH_KERNEL_SOURCE = r"""
#ifdef BACKEND_IS_OPENCL
    #define __device
#endif

#ifdef USE_FLOAT16_BUFFERS
    #ifdef BACKEND_IS_OPENCL
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    #endif

    typedef half enf;
#else
    typedef float enf;
#endif

#define MAX_LOCAL_MEMORY_SIZE (24 * 1024)

__kernel void corr_coeffs(
    __global const int *offsets,
    int offsets_size,
    __global const enf *ref,
    __global const char *ref_mask,
    int ref_size,
    __global const enf *target,
    __global const char *target_mask,
    int target_size,
    __global float *coeffs,
    __global int *match_lens
);

#ifdef USE_LOCAL_MEMORY
    __device void _group_copy_to_local(
        __global const enf *src,
        __global const char *src_mask,
        __local enf *dst,
        __local char *dst_mask,
        int size
    );
#endif

__device void _corr_coeff(
    __global const enf *a,
    __global const char *mask_a,
    #ifdef USE_LOCAL_MEMORY
        __local const enf *b,
        __local const char *mask_b,
    #else
        __global const enf *b,
        __global const char *mask_b,
    #endif
    int size,
    __global float *coeff,
    __global int *match_len
);

__kernel void corr_coeffs(
    __global const int *offsets,
    int offsets_size,
    __global const enf *ref,
    __global const char *ref_mask,
    int ref_size,
    __global const enf *target,
    __global const char *target_mask,
    int target_size,
    __global float *coeffs,
    __global int *match_lens
)
{
    #ifdef USE_LOCAL_MEMORY
        // Caches the target signal in local memory.

        const int TARGET_ITEM_SIZE = sizeof (*target) + sizeof (*target_mask);
        const int LOCAL_CACHE_SIZE = MAX_LOCAL_MEMORY_SIZE / TARGET_ITEM_SIZE;

        __local enf target_local[LOCAL_CACHE_SIZE];
        __local char target_mask_local[LOCAL_CACHE_SIZE];

        _group_copy_to_local(
            target, target_mask, target_local, target_mask_local, min(target_size, LOCAL_CACHE_SIZE)
        );
    #endif

    // Computes the comparison offsets.

    int i = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if (i >= offsets_size) {
        return;
    }

    int offset = offsets[i];

    int ref_offset = max(0, offset);
    int target_offset = max(0, -offset);

    int size = min(ref_size - ref_offset, target_size - target_offset);

    __global const enf *ref_begin = ref + ref_offset;
    __global const char *ref_mask_begin = ref_mask + ref_offset;

    #ifdef USE_LOCAL_MEMORY
        __local const enf *target_begin = target_local + target_offset;
        __local const char *target_mask_begin = target_mask_local + target_offset;
    #else
        __global const enf *target_begin = target + target_offset;
        __global const char *target_mask_begin = target_mask + target_offset;
    #endif

    // Computes the corr. coefficient.

    _corr_coeff(
        ref_begin, ref_mask_begin, target_begin, target_mask_begin, size,
        &coeffs[i], &match_lens[i]
    );
}

#ifdef USE_LOCAL_MEMORY
    __device void _group_copy_to_local(
        __global const enf *src,
        __global const char *src_mask,
        __local enf *dst,
        __local char *dst_mask,
        int size
    )
    {
        int n_items_per_thread = (size + get_local_size(0) - 1) / get_local_size(0);

        int current_begin = get_local_id(0) * n_items_per_thread;
        int current_end = min(size, current_begin + n_items_per_thread);

        for (int i = current_begin; i < current_end; ++i) {
            dst[i] = src[i];
            dst_mask[i] = src_mask[i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif

__device void _corr_coeff(
    __global const enf *a,
    __global const char *mask_a,
    #ifdef USE_LOCAL_MEMORY
        __local const enf *b,
        __local const char *mask_b,
    #else
        __global const enf *b,
        __global const char *mask_b,
    #endif
    int size,
    __global float *coeff,
    __global int *match_len
)
{
    int match_len_local = 0;
    
    // Accumulators for moments
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    float sum_a2 = 0.0f;
    float sum_b2 = 0.0f;
    float sum_ab = 0.0f;

    // SINGLE PASS OPTIMIZATION
    for (size_t i = 0; i < size; ++i) {
        char mask = mask_a[i] & mask_b[i];
        if (mask) {
            float val_a = (float)a[i];
            float val_b = (float)b[i];
            
            sum_a += val_a;
            sum_b += val_b;
            sum_a2 += val_a * val_a;
            sum_b2 += val_b * val_b;
            sum_ab += val_a * val_b;
            match_len_local++;
        }
    }

    if (match_len_local == 0 || (float)match_len_local / size < 0.05f) {
        *coeff = NAN;
        *match_len = 0;
        return;
    }

    float n = (float)match_len_local;
    
    // Pearson Correlation Formula using Raw Moments:
    // (N * Sum(AB) - Sum(A)*Sum(B)) / sqrt([N*Sum(A^2) - (Sum(A))^2] * [N*Sum(B^2) - (Sum(B))^2])
    
    float numerator = n * sum_ab - sum_a * sum_b;
    float var_a = n * sum_a2 - sum_a * sum_a;
    float var_b = n * sum_b2 - sum_b * sum_b;
    
    // Check for constant signal (variance ~ 0)
    if (var_a < 1e-9f || var_b < 1e-9f) {
         *coeff = 0.0f;
    } else {
         *coeff = numerator / sqrt(var_a * var_b);
    }
    *match_len = match_len_local;
}
"""

TINY_MATCH_LEN = timedelta(minutes=1)
MIN_SIM_SCORE = 0.65
ROUGH_HZ = 0.1
ROUGH_MIN_LEN = TINY_MATCH_LEN * 0.75
ROUGH_MIN_SCORE = MIN_SIM_SCORE - 0.1
ROUGH_LOOK_WINDOW = timedelta(seconds=1.0 / ROUGH_HZ * 2.0)
MIN_HILL_CLIMB = timedelta(minutes=10)
BIG_MEM_LIMIT = 64 * 1024 * 1024 

_gpu_ctx = None
_gpu_q = None
_gpu_exe = None
_gpu_float_type = None
_gpu_func = None

def find_common_spots(
    grid_data: WaveData,
    aud_data: WaveData,
    limit: Optional[int] = None,
    stride: timedelta = timedelta(seconds=1),
    engine: CalcMethod = CalcMethod.NUMPY,
    do_deep: bool = False
) -> List[SyncAttempt]:
    if grid_data.real_hz != aud_data.real_hz:
        raise ValueError("Signal frequencies should be identical.")
    if grid_data.nominal_hz != aud_data.nominal_hz:
        raise ValueError("Network frequencies should be identical.")

    hz = grid_data.real_hz

    use_solver = {
        CalcMethod.NUMPY: _do_math_cpu,
        CalcMethod.OPENCL: _do_math_gpu,
    }[engine]

    if do_deep:
        print("  > Mode: Exhaustive Search (skipping approximation step)")
        small_len = math.ceil(TINY_MATCH_LEN.total_seconds() * hz)
        bot_shift = - len(aud_data.samples) + small_len
        top_shift = len(grid_data.samples) - small_len
        hop = math.ceil(stride.total_seconds() * hz)

        try_shifts = np.arange(bot_shift, top_shift, hop)

        try_shifts, sims, lens = _scan_batches(
            use_solver, hz, try_shifts, grid_data.samples, aud_data.samples,
            small_len, MIN_SIM_SCORE,
        )
        return _make_comp_list(
            hz, grid_data.samples, aud_data.samples, try_shifts, sims, lens, limit
        )

    quick_fac = math.floor(hz / ROUGH_HZ)
    quick_hz = hz / quick_fac
    
    grid_small = _shrink_arr(grid_data.samples, quick_fac)
    aud_small = _shrink_arr(aud_data.samples, quick_fac)
    small_min_len = math.ceil(ROUGH_MIN_LEN.total_seconds() * quick_hz)

    s_start = - len(aud_small) + small_min_len
    s_end = len(grid_small) - small_min_len + 1
    quick_hop = math.ceil(stride.total_seconds() * quick_hz)
    quick_shifts = np.arange(s_start, s_end, quick_hop, dtype=np.int32)
    
    quick_shifts, _, _ = _scan_batches(
        use_solver, quick_hz, quick_shifts, grid_small, aud_small,
        small_min_len, ROUGH_MIN_SCORE,
    )

    if len(quick_shifts) < 1:
        return []

    real_min_len = math.ceil(TINY_MATCH_LEN.total_seconds() * hz)
    look_size = math.ceil(ROUGH_LOOK_WINDOW.total_seconds() * hz)
    bot = - len(aud_data.samples) + real_min_len
    top = len(grid_data.samples) - real_min_len

    zones = []
    for q_sh in quick_shifts:
        sh = quick_fac * q_sh
        z_start = max(bot, sh - look_size)
        z_end = min(top, sh + look_size + 1)
        if len(zones) > 0 and z_start <= zones[-1][1]:
            zones[-1][1] = z_end
        else:
            zones.append([z_start, z_end])

    real_hop = math.ceil(stride.total_seconds() * hz)
    shifts = np.concatenate([
        np.arange(z_s, z_e, real_hop)
        for z_s, z_e in zones
    ])

    shifts, sims, lens = _scan_batches(
        use_solver, hz, shifts, grid_data.samples, aud_data.samples,
        real_min_len, MIN_SIM_SCORE,
    )

    return _make_comp_list(
        hz, grid_data.samples, aud_data.samples, shifts, sims, lens, limit
    )

def _shrink_arr(arr: np.ma.masked_array, factor: int) -> np.ma.masked_array:
    return arr[::factor]

def _do_math_cpu(
    shifts: np.ndarray, a: np.ma.masked_array, b: np.ma.masked_array
) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.empty(len(shifts), dtype=np.float32)
    lengths = np.empty(len(shifts), dtype=np.int32)
    for i, sh in enumerate(shifts):
        scores[i], lengths[i] = _calc_one_sim(
            a[max(0, sh):sh + len(b)], b[max(0, -sh):len(a) - sh]
        )
    return scores, lengths

def _calc_one_sim(a: np.ma.masked_array, b: np.ma.masked_array) -> Tuple[float, int]:
    bad_mask = a.mask | b.mask
    ok_count = len(a) - np.sum(bad_mask)
    
    if ok_count < 2:
        return np.nan, ok_count

    good_spots = ~bad_mask
    raw_a = a.data[good_spots]
    raw_b = b.data[good_spots]

    try:
        val = np.corrcoef(raw_a, raw_b)[0, 1]
        return val, ok_count
    except Exception:
        return np.nan, ok_count

def _get_errors(
    shifts: np.ndarray, a: np.ma.masked_array, b: np.ma.masked_array
) -> Tuple[np.ndarray, np.ndarray]:
    errs = np.empty(len(shifts), dtype=np.float32)
    for i, sh in enumerate(shifts):
        errs[i] = _calc_rmse(a[max(0, sh):sh + len(b)], b[max(0, -sh):len(a) - sh])
    return errs

def _calc_rmse(a: np.ma.masked_array, b: np.ma.masked_array) -> float:
    assert len(a) == len(b)
    delta = (a - b).astype(np.float64)
    return np.sqrt(np.mean(delta**2))

def _weed_out(
    hz: float, shifts: np.ndarray, sims: np.ndarray, lens: np.ndarray,
    min_len: int, min_sim: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    good = (sims >= min_sim) & (lens >= min_len)
    return shifts[good].copy(), sims[good].copy(), lens[good].copy()

def _make_comp_list(
    hz: float,
    a: np.ma.masked_array, b: np.ma.masked_array,
    shifts: np.ndarray, sims: np.ndarray, lens: np.ndarray,
    cap: Optional[int],
) -> List[SyncAttempt]:
    win = math.ceil(MIN_HILL_CLIMB.total_seconds() * hz)
    keep = np.full(len(shifts), True)
    i = 0
    while i < len(shifts):
        curr_sh = shifts[i]
        curr_sim = sims[i]
        limit = curr_sh + win
        j = i + 1
        while j < len(shifts) and shifts[j] < limit:
            if sims[j] < curr_sim:
                keep[j] = False
                j += 1
            else:
                keep[i] = False
                break
        i = j
    shifts = shifts[keep]
    sims = sims[keep]
    lens = lens[keep]
    
    ranks = np.clip(sims, 0, 1)
    
    combo = sorted(
        zip(shifts, lens, sims, ranks),
        key=lambda m: m[3], 
        reverse=True,
    )
    
    if cap is not None:
        combo = combo[:cap]
    
    final_picks = []
    
    top_shifts = np.array([m[0] for m in combo], dtype=np.int64)
    if len(top_shifts) > 0:
        top_errs = _get_errors(top_shifts, a, b)
    else:
        top_errs = []

    for i, (sh_val, l_val, sim_val, r_val) in enumerate(combo):
        if r_val > 0:
            final_picks.append(SyncAttempt(
                time_shift=timedelta(seconds=int(sh_val / hz)),
                span=timedelta(seconds=int(l_val / hz)),
                similarity=sim_val,
                error_val=top_errs[i] if len(top_errs) > 0 else 0.0, 
                final_score=r_val,
            ))
    
    return final_picks

def _score_simple(hz, lens, sims, errs):
    return np.clip(sims, 0, 1)

def _scan_batches(
    call_fn, op_hz,
    all_shifts, a_dat, b_dat,
    lim_len, lim_sim,
):
    sh_bits = []
    sim_bits = []
    len_bits = []

    big_chunk = BIG_MEM_LIMIT // np.int32(0).nbytes
    chunk_sz = min(big_chunk, 1000)
    
    info = "Matching (Fine)" if op_hz >= 0.5 else "Matching (Approx)"
    tot = len(all_shifts)
    
    if tot > 0:
        with tqdm(total=tot, desc=info, unit="offsets") as bar:
            for start in range(0, tot, chunk_sz):
                fin = min(tot, start + chunk_sz)
                piece = all_shifts[start:fin]

                low = piece.min()
                high = piece.max()
                
                idx_s = max(0, low)
                idx_e = min(len(a_dat), high + len(b_dat))
                
                if idx_s >= idx_e:
                    p_sims = np.full(len(piece), np.nan, dtype=np.float32)
                    p_lens = np.zeros(len(piece), dtype=np.int32)
                else:
                    sub_a = a_dat[idx_s:idx_e]
                    
                    mod_piece = piece - idx_s
                    
                    p_sims, p_lens = call_fn(mod_piece, sub_a, b_dat)

                piece, p_sims, p_lens = _weed_out(
                    op_hz, piece, p_sims, p_lens,
                    lim_len, lim_sim,
                )

                sh_bits.append(piece)
                sim_bits.append(p_sims)
                len_bits.append(p_lens)
                bar.update(fin - start)
    
    if len(sh_bits) > 0:
        final_sh = np.concatenate(sh_bits)
        final_sim = np.concatenate(sim_bits)
        final_len = np.concatenate(len_bits)
    else:
        final_sh = np.empty((0,))
        final_sim = np.empty((0,))
        final_len = np.empty((0,))

    return final_sh, final_sim, final_len

def _init_gpu_stuff():
    global _gpu_ctx, _gpu_q, _gpu_exe, _gpu_float_type, _gpu_func
    
    if _gpu_ctx is None:
        import pyopencl as cl
        ui.pause_listener(True)
        time.sleep(0.2) 
        try:
            _gpu_ctx = cl.create_some_context()
            _gpu_q = cl.CommandQueue(_gpu_ctx)
            bld_opts = ["-DBACKEND_IS_OPENCL"]
            has_half = all(
                "cl_khr_fp16" in d.extensions.split(" ")
                for d in _gpu_ctx.devices
            )
            if has_half:
                _gpu_float_type = np.float16
                bld_opts.append("-DUSE_FLOAT16_BUFFERS")
            else:
                _gpu_float_type = np.float32

            _gpu_exe = cl.Program(
                _gpu_ctx, MATCH_KERNEL_SOURCE
            ).build(options=bld_opts)
            
            try:
                _gpu_func = cl.Kernel(_gpu_exe, "corr_coeffs")
            except TypeError:
                if hasattr(_gpu_exe, "_prg"):
                     _gpu_func = cl.Kernel(_gpu_exe._prg, "corr_coeffs")
                else:
                    raise
        finally:
            ui.pause_listener(False)

def _do_math_gpu(
    shifts: np.ndarray, a: np.ma.masked_array, b: np.ma.masked_array
):
    import pyopencl as cl
    _init_gpu_stuff()

    shifts_cast = shifts.astype(np.int32) 
    
    a_raw = a.astype(_gpu_float_type).data
    b_raw = b.astype(_gpu_float_type).data
    a_raw[np.isnan(a_raw)] = 0.0
    b_raw[np.isnan(b_raw)] = 0.0

    mask_a_byte = np.logical_not(a.mask).astype(np.int8)
    mask_b_byte = np.logical_not(b.mask).astype(np.int8)

    out_sims = np.empty(len(shifts), dtype=np.float32)
    out_lens = np.empty(len(shifts), dtype=np.int32)

    flag = cl.mem_flags
    buf_shifts = cl.Buffer(_gpu_ctx, flag.READ_ONLY | flag.USE_HOST_PTR, hostbuf=shifts_cast)
    buf_a = cl.Buffer(_gpu_ctx, flag.READ_ONLY | flag.USE_HOST_PTR, hostbuf=a_raw)
    buf_b = cl.Buffer(_gpu_ctx, flag.READ_ONLY | flag.USE_HOST_PTR, hostbuf=b_raw)
    buf_ma = cl.Buffer(_gpu_ctx, flag.READ_ONLY | flag.USE_HOST_PTR, hostbuf=mask_a_byte)
    buf_mb = cl.Buffer(_gpu_ctx, flag.READ_ONLY | flag.USE_HOST_PTR, hostbuf=mask_b_byte)

    buf_sims = cl.Buffer(_gpu_ctx, flag.WRITE_ONLY, out_sims.nbytes)
    buf_lens = cl.Buffer(_gpu_ctx, flag.WRITE_ONLY, out_lens.nbytes)

    block_size = 256
    n_grps = math.ceil(len(shifts) / block_size)

    k_args = [
        buf_shifts, np.int32(len(shifts_cast)),
        buf_a, buf_ma, np.int32(len(a_raw)),
        buf_b, buf_mb, np.int32(len(b_raw)),
        buf_sims, buf_lens,
    ]
    for i, item in enumerate(k_args):
        _gpu_func.set_arg(i, item)
    
    cl.enqueue_nd_range_kernel(
        _gpu_q,
        _gpu_func,
        (block_size * n_grps,),
        (block_size,)
    )

    cl.enqueue_copy(_gpu_q, out_sims, buf_sims)
    cl.enqueue_copy(_gpu_q, out_lens, buf_lens)

    return out_sims, out_lens