#!/usr/bin/env bash
# Reprint a text file every time it changes. Ctrl+C to stop.
#   bash watch/watch_file.sh                 # watches watch/file.txt
#   bash watch/watch_file.sh path/to/other.txt
#   touch watch/file.txt                     # force a redraw / status refresh
#
# file.txt (gitignored) is sections of "▸ label" lines, each followed by an indented path --
# repo-relative for repo files, absolute for anything outside the repo. A path ending in .log
# is annotated with the run's state: a log holding "------ FINISHED ------" is done, anything
# else is still going. A path line may end in "(pod)" or "(local)" -- where the run happened --
# which is printed after the status. Same watcher as ../projects/llm/watch, adapted to that format.
f="${1:-watch/file.txt}"
touch "$f"

status() {  # $1 = a log path
  [[ -f $1 ]] || { echo "-- missing"; return; }
  grep -q -- "------ FINISHED ------" "$1" && echo "-- finished" || echo "-- in progress"
}

where() {  # $1 = a line; prints its "(pod)" / "(local)" tag, if any
  [[ $1 =~ [[:space:]](\((pod|local)\))[[:space:]]*$ ]] && printf '%s' "${BASH_REMATCH[1]}"
}

logpath() {  # $1 = a line; prints the path if the line is a log path
  local p="${1#"${1%%[![:space:]]*}"}"
  p="${p%%[[:space:]]*}"
  [[ $p == *.log ]] && printf '%s' "$p"
}

# Redraw when the STATUSES change, not when the logs do. The logs are rewritten
# every few seconds, so following their timestamps made the terminal blink
# constantly; following the statuses means one redraw when a run finishes.
sig() {
  stat -c %Y.%s "$f" 2>/dev/null
  while IFS= read -r line || [[ -n $line ]]; do
    p=$(logpath "$line") && [[ -n $p ]] && status "$p"
  done < "$f"
}

last=""
while true; do
  now=$(sig)
  if [[ "$now" != "$last" ]]; then
    clear
    echo "── $f  ($(date +%H:%M:%S)) ──"
    while IFS= read -r line || [[ -n $line ]]; do
      p=$(logpath "$line")
      w=$(where "$line")
      [[ -n $w ]] && line="${line%"$w"*}" && line="${line%"${line##*[![:space:]]}"}"
      if [[ -n $p ]]; then
        printf '%s  %s%s\n' "$line" "$(status "$p")" "${w:+ $w}"
      else
        printf '%s%s\n' "$line" "${w:+  $w}"
      fi
    done < "$f"
    last="$now"
  fi
  sleep 0.5
done
