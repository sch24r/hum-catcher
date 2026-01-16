import argparse
import sys
import threading
import numpy as np
from datetime import timedelta, timezone
from typing import Optional

from . import analysis, match, csv_utils, ui
from .types import WaveData
from .match import CalcMethod

def main():
    parser = argparse.ArgumentParser(description="Match media ENF against CSV grid data.")
    
    parser.add_argument("media_file", nargs='?', help="Path to the target media file (audio or video)")
    parser.add_argument("csv_file", nargs='?', help="Path to the reference CSV grid frequency file")
    
    parser.add_argument("--frequency", "-f", type=float, default=50.0, help="Network frequency (default: 50.0)")
    parser.add_argument("--backend", "-b", choices=["numpy", "opencl"], default="numpy", help="Matching backend (default: numpy)")
    parser.add_argument("--max-matches", "-m", type=int, default=10, help="Number of top matches to display")
    parser.add_argument("--plot-match", "-pm", action="store_true", help="Plot the best match result")
    parser.add_argument("--plot-target", "-pt", nargs='?', const=True, help="Plot the extracted target ENF signal")
    parser.add_argument("--plot-reference", "-pr", nargs='?', const=True, help="Plot the reference ENF signal from CSV")
    parser.add_argument("--exhaustive", "-e", action="store_true", help="Perform exhaustive search (slower but more accurate, skips approximation)")
    parser.add_argument("--no-partial", "-np", action="store_true", help="Filter out matches that do not fully overlap with the grid data")
    
    args = parser.parse_args()

    show_target = bool(args.plot_target)
    show_reference = bool(args.plot_reference)
    show_match = args.plot_match
    needs_matching = show_match

    if not (show_target or show_reference or show_match):
        parser.error("At least one action flag is required: -pt (target), -pr (reference), or -pm (match).")
        
    f_media = None
    f_csv = None

    if isinstance(args.plot_target, str):
        f_media = args.plot_target
    if isinstance(args.plot_reference, str):
        f_csv = args.plot_reference

    pos_args = [p for p in [args.media_file, args.csv_file] if p is not None]
    
    media_req = show_target or needs_matching
    
    if media_req and f_media is None and len(pos_args) > 0:
        f_media = pos_args.pop(0)

    csv_req = show_reference or needs_matching
    
    if csv_req and f_csv is None and len(pos_args) > 0:
        f_csv = pos_args.pop(0)

    if not f_media and not f_csv:
        parser.print_help()
        sys.exit(0)

    target_wave = None
    ref_wave = None

    if f_media:
        if show_target or needs_matching:
            print(f"Loading and analyzing media: {f_media}...")
            try:
                raw_aud, rate = analysis.grab_audio(f_media)
                result = analysis.dig_for_hum(raw_aud, rate, grid_target=args.frequency)
                
                if result is None:
                    print("Error: No ENF signal detected in the audio file.")
                    if needs_matching: sys.exit(1)
                else:
                    target_wave = result.hum_wave
                    print(f"  > Detected ENF signal (Quality: {target_wave.get_quality():.2%}, Duration: {target_wave.total_time})")
                    
                    valid_part = target_wave.samples.compressed() + target_wave.nominal_hz
                    if len(valid_part) > 0:
                        print(f"  > Target Stats: Min={np.min(valid_part):.3f}Hz Max={np.max(valid_part):.3f}Hz Med={np.median(valid_part):.3f}Hz Mean={np.mean(valid_part):.3f}Hz")

                    if show_target:
                        ui.draw_wave(target_wave, "plot_target.png", "Target ENF Signal")
            except Exception as e:
                print(f"Error reading media: {e}")
                if needs_matching: sys.exit(1)
    elif show_target:
        print("Error: --plot-target requires a media file argument or positional media argument.")
        sys.exit(1)

    if f_csv:
        if show_reference or needs_matching:
            settings = csv_utils.setup_csv(f_csv)
            
            print(f"\nLoading grid data from CSV: {f_csv}...")
            print("  (Press 'q' to abort processing at any time)")
            
            watcher = threading.Thread(target=ui.watch_keys, daemon=True)
            watcher.start()

            grid_raw = csv_utils.ingest_csv(f_csv, settings, args.frequency)
            
            if len(grid_raw[0]) == 0:
                print("Error: No valid data found in CSV.")
                if needs_matching: sys.exit(1)
            else:
                goal_hz = target_wave.real_hz if target_wave else 1.0

                filled_vals, t_start = csv_utils.fill_gaps(
                    grid_raw, 
                    goal_hz 
                )
                
                ref_wave = WaveData(
                    nominal_hz=args.frequency,
                    real_hz=goal_hz,
                    samples=filled_vals,
                    start_time=t_start
                )
                
                tz_disp = settings.get('timezone', timezone.utc)
                
                if ref_wave.start_time:
                    disp_s = ref_wave.start_time.astimezone(tz_disp)
                    disp_e = ref_wave.end_time.astimezone(tz_disp)
                    print(f"  > Grid signal loaded (Start: {disp_s}, End: {disp_e}, Duration: {ref_wave.total_time})")
                else:
                     print(f"  > Grid signal loaded (Duration: {ref_wave.total_time})")

                valid_grid = ref_wave.samples.compressed() + ref_wave.nominal_hz
                if len(valid_grid) > 0:
                    print(f"  > Reference Stats: Min={np.min(valid_grid):.3f}Hz Max={np.max(valid_grid):.3f}Hz Med={np.median(valid_grid):.3f}Hz Mean={np.mean(valid_grid):.3f}Hz")

                if show_reference:
                    ui.draw_wave(ref_wave, "plot_reference.png", "Reference ENF Signal")
    elif show_reference:
        print("Error: --plot-reference requires a CSV file argument or positional CSV argument.")
        sys.exit(1)

    if needs_matching:
        if not target_wave:
            print("Error: Cannot match without valid media ENF signal.")
            sys.exit(1)
        if not ref_wave:
            print("Error: Cannot match without valid grid reference signal.")
            sys.exit(1)
            
        print("\nMatching signals...")
        engine_val = CalcMethod(args.backend)
        
        found = match.find_common_spots(
            ref_wave, 
            target_wave, 
            limit=args.max_matches, 
            engine=engine_val,
            do_deep=args.exhaustive
        )

        if args.no_partial:
            start_cnt = len(found)
            found = [
                x for x in found 
                if x.time_shift >= timedelta(0) and (x.time_shift + target_wave.total_time) <= ref_wave.total_time
            ]
            dropped = start_cnt - len(found)
            if dropped > 0:
                print(f"  > Note: {dropped} matches were found but filtered out by --no-partial.")

        if not found:
            print("No matches found.")
        else:
            if 'tz_disp' not in locals():
                tz_disp = timezone.utc
            
            tz_name = str(tz_disp)
            print(f"\nFound {len(found)} matches:")
            print(f"{f'START TIME ({tz_name})':<35} {f'END TIME ({tz_name})':<35} {'START (UNIX)':<15} {'END (UNIX)':<15} {'CORR':<10} {'RMSE':<10}")
            print("-" * 120)
            for m in found:
                t_utc = ref_wave.start_time + m.time_shift
                e_utc = t_utc + target_wave.total_time
                disp_t = t_utc.astimezone(tz_disp)
                disp_e = e_utc.astimezone(tz_disp)
                u_start = t_utc.timestamp()
                u_end = e_utc.timestamp()
                
                print(f"{str(disp_t):<35} {str(disp_e):<35} {u_start:<15.3f} {u_end:<15.3f} {m.similarity:<10.4f} {m.error_val:<10.4f}")

            if show_match:
                print("\nPlotting matches...")
                for i, m in enumerate(found):
                    t_utc = ref_wave.start_time + m.time_shift
                    disp_t = t_utc.astimezone(tz_disp)
                    ts_txt = disp_t.strftime("%Y-%m-%d_%H-%M-%S")
                    fname = f"match_{i+1}_{ts_txt}.png"
                    head = f"Match {i+1}: {disp_t} (Corr: {m.similarity:.4f})"
                    ui.draw_fit(m, ref_wave, target_wave, fname, head)

if __name__ == "__main__":
    main()
