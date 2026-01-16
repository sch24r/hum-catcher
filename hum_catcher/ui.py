import sys
import os
import time
import select
import atexit
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

try:
    from dateutil import parser as user_date_parser
    HAS_UDATE = True
except ImportError:
    HAS_UDATE = False

if os.name == 'posix':
    import termios

_KEY_STOPPED = False
_OLD_TERM = None

def pause_listener(is_paused: bool):
    global _KEY_STOPPED
    _KEY_STOPPED = is_paused

def get_text(msg: str) -> str:
    try:
        txt = input(msg)
    except EOFError:
        sys.exit(1)
    if txt.strip().lower() in ['q', 'quit', 'exit']:
        print("\n  > User requested exit. Bye!")
        sys.exit(0)
    return txt

def get_yes_no(msg: str, std_val: bool = True) -> bool:
    hint = "Y/n" if std_val else "y/N"
    while True:
        ans = get_text(f"{msg} [{hint}]: ").strip().lower()
        if not ans:
            return std_val
        if ans in ['y', 'yes']:
            return True
        if ans in ['n', 'no']:
            return False
        print("  > Error: Please enter 'y' or 'n'.")

def get_zone_info() -> timezone:
    while True:
        u_zone = get_text("  > Please enter timezone (e.g., 'UTC', 'Europe/Berlin', '+0100') [default: UTC]: ").strip()
        if not u_zone:
            print("  > Defaulting to UTC.")
            return timezone.utc
        if ZoneInfo:
            try:
                return ZoneInfo(u_zone)
            except Exception:
                pass
        if re.match(r'^[+-]\d{4}$', u_zone):
            try:
                return datetime.strptime(u_zone, "%z").tzinfo
            except ValueError:
                pass
        if u_zone.upper() == 'UTC':
            return timezone.utc
        print(f"  > Error: Could not understand timezone '{u_zone}'.")
        if not ZoneInfo:
            print("  > (Note: ZoneInfo not available, try offsets like +0100)")

def _make_timeline(t_start, t_rate, count, use_nums=False):
    if t_start and not use_nums:
        return [t_start + i * t_rate for i in range(count)]
    else:
        root = t_start if t_start else timedelta(0)
        return [(root + i * t_rate).total_seconds() for i in range(count)]

def draw_fit(hit, grid, aud, f_name: str, caption: str):
    import math
    import numpy as np
    
    if grid.real_hz != aud.real_hz:
        raise ValueError("Signal frequencies should be identical.")
    
    amount = min(grid.total_time - hit.time_shift, aud.total_time)
    pt_count = math.floor(amount.total_seconds() * grid.real_hz)
    
    time_pts = _make_timeline(grid.start_time + hit.time_shift if grid.start_time else hit.time_shift, 
                          grid.step_time, pt_count, use_nums=grid.start_time is None)

    skip_pts = math.floor(hit.time_shift.total_seconds() * grid.real_hz)
    grid_vals = np.ma.empty(pt_count, dtype=np.float32)
    grid_vals[:] = np.ma.masked
    g_start = max(0, skip_pts)
    pts_copy = min(len(grid.samples) - g_start, pt_count)
    if pts_copy > 0:
        grid_vals[0:pts_copy] = grid.samples[g_start:g_start + pts_copy]
    aud_vals = aud.samples[0:pt_count].astype(np.float32)

    _render_chart(time_pts, [
        (grid_vals + grid.nominal_hz, "blue", "Reference ENF"),
        (aud_vals + aud.nominal_hz, "red", "Target ENF")
    ], f_name, caption, fancy_date=bool(grid.start_time))

def draw_wave(stuff, f_name: str, caption: str):
    x_axis = _make_timeline(stuff.start_time, stuff.step_time, len(stuff.samples), use_nums=stuff.start_time is None)
    _render_chart(x_axis, [
        (stuff.samples + stuff.nominal_hz, "green", "Signal")
    ], f_name, caption, fancy_date=bool(stuff.start_time))

def _render_chart(x, lines, f_out, top_txt, fancy_date=False):
    plt.figure(figsize=(12, 6))
    for vals, ink, tag in lines:
        plt.plot(x, vals, color=ink, label=tag, linewidth=1)
    plt.title(top_txt)
    plt.xlabel("Time")
    plt.ylabel("Frequency (Hz)")
    if len(lines) > 1: plt.legend()
    plt.grid(True, alpha=0.3)
    if fancy_date: plt.gcf().autofmt_xdate()
    print(f"  > Saving plot to {f_out}...")
    plt.savefig(f_out)
    plt.close()

def _fix_term():
    if _OLD_TERM and sys.stdin.isatty():
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _OLD_TERM)

def _prep_term(handle):
    flavor = termios.tcgetattr(handle)
    flavor[3] = flavor[3] & ~(termios.ECHO | termios.ICANON)
    mode_opts = flavor[6]
    mode_opts[termios.VMIN] = 1
    mode_opts[termios.VTIME] = 0
    termios.tcsetattr(handle, termios.TCSANOW, flavor)

def watch_keys():
    global _OLD_TERM
    if os.name != 'posix' or not sys.stdin.isatty():
        return
    handle = sys.stdin.fileno()
    try:
        if _OLD_TERM is None:
            _OLD_TERM = termios.tcgetattr(handle)
            atexit.register(_fix_term)
        raw_mode = False
        while True:
            if _KEY_STOPPED:
                if raw_mode:
                    termios.tcsetattr(handle, termios.TCSADRAIN, _OLD_TERM)
                    raw_mode = False
                time.sleep(0.1)
                continue
            if not raw_mode:
                _prep_term(handle)
                raw_mode = True
            
            if select.select([sys.stdin], [], [], 0.1)[0]:
                if _KEY_STOPPED:
                    continue
                btn = sys.stdin.read(1)
                if btn.lower() == 'q':
                    _fix_term()
                    print("\n\n  > 'q' pressed. Aborting...")
                    os._exit(0)
    except Exception:
        _fix_term()