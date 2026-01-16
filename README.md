# Hum Catcher

A robust command-line tool for **Electrical Network Frequency (ENF)** analysis. This tool automates the extraction of ENF signals (mains hum) from audio/video recordings and matches them against historical power grid frequency databases to pinpoint the exact time of recording.

## Features

- **Robust Audio Analysis**: Uses multi-harmonic combination and STFT signal processing to extract high-quality ENF signals even from noisy audio.
- **Media Support**: Supports various audio and video formats.
- **Interactive CSV Import Wizard**:
  - Auto-detects delimiters and headers.
  - Supports combined Date/Time columns, separate columns, and Unix timestamps.
  - Interactive timezone configuration (supports automatic detection, `zoneinfo`, and manual offsets).
  - Handles real-world grid data, including sparse datasets, by resampling to match the audio.
- **Advanced Matching Engine**:
  - **Dual Backend**: Supports **NumPy** (CPU) and **OpenCL** (GPU) for accelerated matching.
  - **Exhaustive Search**: Optional brute-force mode for maximum precision.
  - **Approximation**: Defaults to a faster decimated search followed by fine-tuning ("coarse-to-fine" strategy).
  - **Partial Matching**: Configurable filtering for matches that extend beyond available grid data.
- **User Experience**:
  - **Visualizations**: Generates plots comparing the reference and target signals.
  - **Progress Feedback**: Detailed progress bars for file parsing, resampling, and matching.
  - **Graceful Exit**: Press `q` at any safe point to abort operations.

## Prerequisites

- **Python**: 3.8+ (3.9+ recommended for built-in timezone support).
- **System Tool**: [FFmpeg](https://ffmpeg.org/) must be installed and available in your system PATH for media processing.

## Usage

You can run the tool as a python module from the source root:

```bash
python -m hum_catcher <media_file> <csv_file> [options]
```

### Arguments

| Argument | Description |
| :--- | :--- |
| `media_file` | Path to the target media file (audio or video). |
| `csv_file` | Path to the reference grid frequency CSV file. |
| `-f`, `--frequency` | Network frequency (e.g., `50` for EU/Asia, `60` for US). Default: `50.0`. |
| `-b`, `--backend` | Computation backend: `numpy` (default) or `opencl` (requires GPU setup). |
| `-m`, `--max-matches` | Number of top matches to display. Default: `10`. |
| `-pm`, `--plot-match` | Plot the best match result. |
| `-pt`, `--plot-target` | Plot the extracted target ENF signal. |
| `-pr`, `--plot-reference` | Plot the reference ENF signal from CSV. |
| `-e`, `--exhaustive` | Skip approximation steps and perform a exhaustive search (slower but potentially more accurate). |
| `-np`, `--no-partial` | Filter out matches that do not fully fit within the grid data duration. |

### Usage Examples

**Analyze Audio Only (No Matching):**
```bash
python -m hum_catcher evidence.wav -pt
```

**Matching Search:**
```bash
python -m hum_catcher evidence.wav grid_2025.csv -pm
```

**US Grid Search (60Hz Grid):**
```bash
python -m hum_catcher recording.mp4 us_grid_data.csv -f 60 -pm
```

**High Performance GPU Search:**
```bash
python -m hum_catcher large_recording.flac huge_grid_db.csv --backend opencl --exhaustive -pm
```

## The Interactive Wizard

When you start the tool, it analyzes the media file first. Then, it loads the CSV file. Since grid data formats vary wildly, Hum Catcher will ask you to confirm details about the CSV structure in the terminal:

1.  **Header & Delimiter**: It attempts to sniff the CSV format (`,`, `;`, `\t`) and header presence.
2.  **Column Selection**: You will be asked to identify which columns contain the Timestamp and Frequency data.
3.  **Time Format**: It supports standard date strings (e.g., `2023-10-27 14:30:00`) or Unix Timestamps.
4.  **Timezone**: It attempts to detect timezones from the data. If ambiguous, you can input a region (e.g., `Europe/Berlin`) or offset (e.g., `+0100`).

## Understanding Results

The output table provides the following customized metrics:

- **START/END (Local/Unix)**: The calculated timestamp of the recording based on the grid match.
- **CORR (Correlation)**: Pearson correlation coefficient (0.0 to 1.0). Indicates shape similarity. Higher is better.
- **RMSE**: Root Mean Square Error. Indicates magnitude similarity.

## Troubleshooting

- **"No ENF signal detected"**: The recording might be digital silence, too short, or lack conductive electrical hum.
- **OpenCL Errors**: If using the `opencl` backend, ensure you have the correct drivers (CUDA/ROCm/Intel Compute) installed for `pyopencl` to access your GPU.
