#!/bin/sh
# Opens the MS2 app. It runs on the lab server over ssh -- or right here when HOST=localhost.
# Settings are in ms2_settings.txt next to this file. Ctrl+C (or closing the terminal) stops
# the app; analyses that are running keep going on the server.
cd "$(dirname "$0")" || exit 1
get() { sed -n "s/^$1=//p" ms2_settings.txt | tr -d '\r'; }
HOST=$(get HOST); SSH_USER=$(get SSH_USER); PORT=$(get PORT); MS2=$(get MS2); DATA_ROOT=$(get DATA_ROOT)
URL="http://localhost:$PORT/"

# Open the browser as soon as the app answers.
(
  for _ in $(seq 90); do
    if curl -s -o /dev/null "$URL"; then xdg-open "$URL" >/dev/null 2>&1; exit; fi
    sleep 1
  done
) &

if [ "$HOST" = localhost ]; then
  "$MS2" app --port "$PORT" --data-root "$DATA_ROOT"
else
  echo "Connecting to $HOST as $SSH_USER ..."
  ssh -t -o ExitOnForwardFailure=yes -L "$PORT:localhost:$PORT" "$SSH_USER@$HOST" \
      "$MS2 app --port $PORT --data-root $DATA_ROOT"
fi
echo "The MS2 app has stopped. Analyses that were running keep going on the server."
