"""One Dream-1 trial in Dream YAM Sim: a task in words, a cap on paid calls, a video and a score.

Runs in envs/dream_yam/.venv; the station (sim, request code, referee) is
dream_robot/sims/dream_yam/station.py.

    envs/dream_yam/.venv/bin/python experiments/dream1/scripts/trial.py \
        --task "put apple into the bowl" --calls 50 --out scratch/dream1/apple_bowl

Writes cameras.mp4, log.json (calls, start/end score), trace.json (per step: grip command,
opening, object heights, fingertips on the apple) and chunks.json (every chunk played).
The key: DREAM_API_KEY, else DREAM_LABS_API_KEY from the repo's .env. Never printed.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio

from dream_robot.sims.dream_yam.station import MODEL, REPO, Station


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="put apple into the bowl")
    parser.add_argument("--calls", type=int, default=25, help="paid get_action calls, at most")
    parser.add_argument("--rows", type=int, default=48, help="rows played per call, 1-48")
    parser.add_argument("--out", type=Path, default=REPO / "scratch/dream1/trial")
    parser.add_argument("--dry-run", action="store_true", help="everything but the API call")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    station = Station()
    log = {"task": args.task, "model": MODEL, "start": station.score()}
    try:
        if args.dry_run:
            station.hold(36)
            print("dry run OK: station steps and records, no API call")
        else:
            log.update(station.run_policy(args.task, args.calls, args.rows,
                                          log=lambda s: print(s, flush=True)))
    finally:
        with imageio.get_writer(args.out / "cameras.mp4", fps=20, macro_block_size=8) as video:
            for frame in station.take_frames():
                video.append_data(frame)
        log["end"] = station.score()
        (args.out / "log.json").write_text(json.dumps(log, indent=2))
        (args.out / "trace.json").write_text(json.dumps(station.trace))
        (args.out / "chunks.json").write_text(json.dumps(station.chunks))
        station.close()
        moved = {k: v for k, v in log["end"].items() if v["in_bowl"] or v["on_plate"]}
        print(f"calls {log.get('calls', 0)} · sim {station.steps * 0.05:.1f}s · "
              f"in bowl / on plate: {moved or 'nothing'}")
        print(f"video: {args.out / 'cameras.mp4'}")


if __name__ == "__main__":
    main()
