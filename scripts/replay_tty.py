import time
import os
import sys
import select
import termios
import struct
import argparse
import glob


def replay_tty_frames(input_file, speed=0.2):
    """Replay TTY frames from a given input file at a given playback speed (default is 0.2)."""
    # save original terminal settings
    old = termios.tcgetattr(sys.stdin.fileno())
    new = termios.tcgetattr(sys.stdin.fileno())
    # configure terminal for raw input
    new[3] &= ~(termios.ICANON | termios.ECHO | termios.ECHONL)
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, new)

    try:
        with open(input_file, "rb") as f:
            f.seek(0, 0)
            frames = []
            # read all frames from the input file
            while True:
                header = f.read(13)
                if not header:
                    break
                sec, usec, length, _ = struct.unpack("<iiiB", header)
                timestamp = sec + usec * 1e-6
                data = f.read(length)
                frames.append((timestamp, data))

            total_frames = len(frames)
            state = {"playing": True, "step": 0}
            frame_number = 0
            prev_timestamp = None
            drift = 0.0

            # initial status
            update_status_display(speed, frame_number, total_frames)

            # playback loop
            while 0 <= frame_number < total_frames:
                timestamp, data = frames[frame_number]

                if prev_timestamp is not None:
                    # Handle playback control and timing
                    speed, state, drift = handle_playback_control(
                        timestamp - prev_timestamp,
                        speed,
                        state,
                        drift,
                    )

                # display current frame
                sys.stdout.write("\033[H")  # move cursor to top-left
                sys.stdout.buffer.write(data)
                sys.stdout.flush()

                prev_timestamp = timestamp

                # update frame number
                if state["playing"] or state["step"] != 0:
                    frame_number += 1 if state["playing"] else state["step"]
                    state["step"] = 0

                update_status_display(speed, frame_number, total_frames)
    finally:
        # restore original terminal settings
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, old)


def update_status_display(speed, frame_number, total_frames):
    """Update the status display with current playback information."""
    sys.stdout.write("\033[H")  # move cursor to top-left
    sys.stdout.write("\033[K")  # clear line
    status = f"SPEED: {speed:.2f}x | FRAME: {frame_number}/{total_frames}"
    sys.stdout.write(status)
    sys.stdout.flush()


def handle_playback_control(diff, speed, state, drift=0.0):
    """Handle playback control, including speed changes and frame navigation."""
    start = time.time()

    commands = {
        b"+": lambda s: min(s * 2, 16),  # double speed, max 16x
        b"-": lambda s: max(s / 2, 0.25),  # halve speed, min 0.25x
        b"1": lambda s: 1,  # reset to normal speed (1x)
        b" ": lambda s: s,  # toggle play/pause
    }

    rlist, _, _ = select.select([sys.stdin], [], [], 0)
    if rlist:
        c = os.read(sys.stdin.fileno(), 3)  # read up to 3 bytes for arrow keys
        if c in commands:
            new_speed = commands[c](speed)
            if c == b" ":
                state["playing"] = not state["playing"]
            elif new_speed != speed:
                speed = new_speed
        elif c == b"\x1b[C" and not state["playing"]:  # right arrow
            # step forward
            state["step"] = 1
        elif c == b"\x1b[D" and not state["playing"]:  # left arrow
            # step backward
            state["step"] = -1
        drift = 0.0

    if state["playing"]:
        # calculate wait time for consistent playback speed
        wait_time = max((diff / speed) - drift, 0.0)
        time.sleep(wait_time)
        stop = time.time()
        drift = (stop - start) - wait_time
    elif state["step"] == 0:
        # wait for next input
        select.select([sys.stdin], [], [])
        drift = 0.0

    return speed, state, drift


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
