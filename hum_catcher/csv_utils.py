import sys
import csv
import numpy as np
import scipy.interpolate
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from . import ui

try:
    from dateutil import parser as user_date_parser
    HAS_UDATE = True
except ImportError:
    HAS_UDATE = False

def _try_parse_date(txt):
    if HAS_UDATE:
        try:
            return user_date_parser.parse(txt)
        except Exception:
            pass
    
    attempts = [
        "%d.%m.%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ", "%d.%m.%Y %H:%M",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in attempts:
        try:
            return datetime.strptime(txt, fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse datetime: {txt}")

def setup_csv(loc):
    print(f"\n--- CSV Configuration: {loc} ---")
    print("  (Type 'q', 'quit' or 'exit' at any prompt to exit)")
    try:
        with open(loc, 'r', encoding='utf-8') as fh:
            chunk = fh.read(4096)
            fh.seek(0)
            rows_tot = sum(1 for _ in fh)
            fh.seek(0)
            lines = []
            for _ in range(5):
                row = fh.readline()
                if not row: break
                lines.append(row.rstrip('\n'))
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    print(f"  > Total Rows: {rows_tot}")
    print("\n  > Raw File Content (First 5 lines):")
    for i, l in enumerate(lines):
        print(f"    Line {i}: {l}")

    try:
        sniffed = csv.Sniffer().sniff(chunk)
        std_sep = sniffed.delimiter
        has_head = csv.Sniffer().has_header(chunk)
    except csv.Error:
        std_sep = ';'
        has_head = True

    print(f"\n  > Detected Header: {'Yes' if has_head else 'No'}")
    use_head = ui.get_yes_no("  > Does the CSV have a header?", std_val=has_head)

    print(f"\n  > Detected Separator: '{std_sep}'")
    usr_sep = ui.get_text(f"  > Enter separator (e.g. , or ; or \\t) [default: '{std_sep}']: ")
    if usr_sep == '\\t' or usr_sep.lower() == 'tab':
        sep = '\t'
    elif usr_sep:
        sep = usr_sep
    else:
        sep = std_sep
        
    if len(sep) != 1:
        print(f"  > Warning: Separator '{sep}' is invalid. Using default '{std_sep}'.")
        sep = std_sep

    data_snippet = []
    with open(loc, 'r', encoding='utf-8') as fh:
        rdr = csv.reader(fh, delimiter=sep)
        for i, r in enumerate(rdr):
            if i >= 5: break
            data_snippet.append(r)
            
    cols_cnt = len(data_snippet[0]) if data_snippet else 0
    print(f"\n  > Parsed Data Preview (using separator '{sep}'):")
    print(f"  > Detected Columns: {cols_cnt}")
    for i, r in enumerate(data_snippet):
        print(f"    Row {i}: {r}")

    print("\n  > Select Date/Time Format:")
    print("    1. Date and Time in the SAME column (e.g., '2025-01-01 12:00:00' or Unix Timestamp)")
    print("    2. Date and Time in SEPARATED columns")
    while True:
        sel = ui.get_text("  > Choice [1/2]: ").strip()
        if sel in ['1', '2']:
            break
        print("  > Error: Please enter '1' or '2'.")

    settings = {
        'has_header': use_head,
        'delimiter': sep,
        'mode': 'combined' if sel == '1' else 'separated',
        'total_rows': rows_tot
    }
    
    def ask_idx(msg):
        while True:
            try:
                txt_v = ui.get_text(msg)
                v = int(txt_v)
                if 0 <= v < cols_cnt:
                    return v
                print(f"    Error: Column index must be between 0 and {cols_cnt-1}")
            except ValueError:
                print(f"    Error: Please enter a number between 0 and {cols_cnt-1}.")

    if settings['mode'] == 'combined':
        settings['dt_col'] = ask_idx("  > Enter Column Index for Date/Time (0-based): ")
        settings['freq_col'] = ask_idx("  > Enter Column Index for Frequency (0-based): ")
        
        looks_unix = False
        try:
            chk_i = 1 if use_head and len(data_snippet) > 1 else 0
            if len(data_snippet) > chk_i:
                sample = data_snippet[chk_i][settings['dt_col']].strip().strip('"\'')
                f_val = float(sample)
                if 100_000_000 < f_val < 10_000_000_000:
                    looks_unix = True
        except (ValueError, IndexError):
            pass

        if looks_unix:
            print("  > Detected likely Unix timestamp format.")
        settings['is_unix'] = ui.get_yes_no("  > Is this a Unix/Epoch timestamp?", std_val=looks_unix)
    else:
        settings['date_col'] = ask_idx("  > Enter Column Index for Date (0-based): ")
        settings['time_col'] = ask_idx("  > Enter Column Index for Time (0-based): ")
        settings['freq_col'] = ask_idx("  > Enter Column Index for Frequency (0-based): ")
        settings['is_unix'] = False

    demo_row = data_snippet[1] if use_head and len(data_snippet) > 1 else data_snippet[0]
    found_tz = None
    if not settings['is_unix']:
        try:
            if settings['mode'] == 'combined':
                str_dt = demo_row[settings['dt_col']]
            else:
                str_dt = f"{demo_row[settings['date_col']]} {demo_row[settings['time_col']]}"
            parsed = _try_parse_date(str_dt)
            if parsed.tzinfo is not None:
                found_tz = parsed.tzinfo
        except Exception:
            pass

    if found_tz:
        print(f"\n  > Detected Timezone in data: {found_tz}")
        if ui.get_yes_no("  > Use this timezone?", std_val=True):
            settings['timezone'] = found_tz
        else:
            settings['timezone'] = ui.get_zone_info()
    elif settings['is_unix']:
        print("\n  > Unix timestamp detected.")
        settings['timezone'] = ui.get_zone_info()
    else:
        print("\n  > No timezone detected in data.")
        settings['timezone'] = ui.get_zone_info()
    return settings

def ingest_csv(loc, settings, base_hz=50.0):
    tot = settings['total_rows']
    t_vals = np.empty(tot, dtype=np.float64)
    hz_vals = np.empty(tot, dtype=np.float32)
    
    cnt = 0

    with open(loc, 'r', encoding='utf-8') as fh:
        rdr = csv.reader(fh, delimiter=settings['delimiter'])
        if settings['has_header']:
            next(rdr, None)

        for row in tqdm(rdr, total=tot, desc="Parsing CSV", unit="rows"):
            if not row: continue
            try:
                txt_hz = row[settings['freq_col']]
                val_hz = float(txt_hz.replace(',', '.'))
                diff = val_hz - base_hz
                final_dt = None
                if settings['mode'] == 'combined':
                    txt = row[settings['dt_col']]
                    if settings['is_unix']:
                        stamp = float(txt)
                    else:
                        final_dt = _try_parse_date(txt)
                        if final_dt.tzinfo is None and settings['timezone']:
                            final_dt = final_dt.replace(tzinfo=settings['timezone'])
                        stamp = final_dt.timestamp()
                else:
                    d_txt = row[settings['date_col']]
                    t_txt = row[settings['time_col']]
                    full_txt = f"{d_txt} {t_txt}"
                    final_dt = _try_parse_date(full_txt)
                    if final_dt.tzinfo is None and settings['timezone']:
                        final_dt = final_dt.replace(tzinfo=settings['timezone'])
                    stamp = final_dt.timestamp()
                
                t_vals[cnt] = stamp
                hz_vals[cnt] = diff
                cnt += 1
            except (ValueError, IndexError):
                continue
    
    t_vals = t_vals[:cnt]
    hz_vals = hz_vals[:cnt]
    
    order = np.argsort(t_vals)
    return t_vals[order], hz_vals[order]

def fill_gaps(raw_data, goal_hz):
    ts, vs = raw_data
    
    if len(ts) == 0:
         return np.ma.empty(0), None

    start_ts = ts[0]
    dt_obj_start = datetime.fromtimestamp(start_ts, tz=timezone.utc)

    def nice_time(dt: datetime) -> datetime:
        if dt.microsecond < 500_000:
            return dt.replace(microsecond=0)
        else:
            return dt.replace(microsecond=0) + timedelta(seconds=1)

    clean_start = nice_time(dt_obj_start)
    end_ts = ts[-1]
    
    step = 1.0 / goal_hz
    
    offset_val = start_ts - clean_start.timestamp() 
    
    span = end_ts - clean_start.timestamp()
    count = int(span * goal_hz)
    
    src_x = ts - clean_start.timestamp()
    src_y = vs
    
    mode = 'cubic' if len(src_x) > 3 else 'linear'
    mapper = scipy.interpolate.interp1d(
        src_x, src_y, kind=mode, bounds_error=False, fill_value=np.nan,
        copy=False, assume_sorted=True
    )
    
    out_y = np.empty(count, dtype=np.float32)
    chunk = 1_000_000
    
    with tqdm(total=count, desc="Resampling Grid", unit="samples") as bar:
        for i in range(0, count, chunk):
            lim = min(i + chunk, count)
            x_part = np.arange(i, lim, dtype=np.float64) * step
            out_y[i:lim] = mapper(x_part)
            bar.update(lim - i)

    bad_gap = 4.0 * step
    diffs = np.diff(src_x)
    holes = np.where(diffs > bad_gap)[0]
    
    if len(holes) > 0:
        margin = 2.0 * step
        for i in holes:
            g_s = src_x[i] + margin
            g_e = src_x[i+1] - margin
            idx_s = int(np.ceil(g_s * goal_hz))
            idx_e = int(np.floor(g_e * goal_hz))
            idx_s = max(0, idx_s)
            idx_e = min(count, idx_e)
            if idx_s < idx_e:
                out_y[idx_s:idx_e] = np.nan

    mask = np.isnan(out_y)
    final = np.ma.masked_array(out_y, mask=mask)
    return final, clean_start