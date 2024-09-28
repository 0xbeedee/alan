#!/usr/bin/env python3

import time
import os
import sys
import select
import termios
import struct
import argparse
import glob


def replay_tty_frames(input_file, speed=0.2):
    # save the original terminal settings
    old = termios.tcgetattr(sys.stdin.fileno())
    new = termios.tcgetattr(sys.stdin.fileno())
    # we want only raw input (no buffering, no echo)
    new[3] &= ~(termios.ICANON | termios.ECHO | termios.ECHONL)
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, new)
    try:
        with open(input_file, "rb") as f:
            prev_timestamp = None
            drift = 0.0
            clear_screen()
            while True:
                # header of each frame is 13 bytes long
                header = f.read(13)
                if not header:
                    break  # EOF
                # seconds, microseconds, data length, and channel
                sec, usec, length, _ = struct.unpack("<iiiB", header)
                timestamp = sec + usec * 1e-6  # seconds
                data = f.read(length)
                if prev_timestamp is not None:
                    speed, drift = handle_playback_control(
                        timestamp - prev_timestamp, speed, drift
                    )
                # move cursor to top-left before writing each frame
                sys.stdout.write("\033[H")
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
                prev_timestamp = timestamp
    finally:
        # restore original terminal settings
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, old)


def handle_playback_control(diff, speed, drift=0.0):
    # TODO more options here? (pausing, re-winding, etc)
    start = time.time()
    # calculate wait time, adjusting for playback speed and previous drift
    diff = max((diff / speed) - drift, 0.0)
    # wait for input or timeout
    rlist, _, _ = select.select([sys.stdin], [], [], diff)
    if rlist:
        # adjust playback speed
        c = os.read(0, 1)
        if c in (b"+", b"f"):
            speed *= 2  # double speed
        elif c in (b"-", b"s"):
            speed /= 2  # halve speed
        elif c == b"1":
            speed = 1  # normal speed
        elif c == b" ":
            select.select([sys.stdin], [], [])
        drift = 0.0
    else:
        # calculate drift (difference between intended and actual wait time)
        stop = time.time()
        drift = (stop - start) - diff
    return speed, drift


def clear_screen():
    # clear screen and move cursor to top-left
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def get_latest_file(directory):
    pattern = os.path.join(directory, "**", "*.ttyrec")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay TTY recordings")
    parser.add_argument(
        "path",
        nargs="?",
        help="The path to the input file or the directory containing the TTY recordings (in which case the most recent recording will be played)",
    )
    args = parser.parse_args()

    path = os.path.expanduser(args.path)

    if os.path.isdir(path):
        input_file = get_latest_file(path)
        if not input_file:
            print(f"[+] No .ttyrec files found in {path}!")
            sys.exit(1)
    elif os.path.isfile(path):
        input_file = path
    else:
        print(f"[+] Invalid path: {path}.")
        sys.exit(1)

    replay_tty_frames(input_file)
