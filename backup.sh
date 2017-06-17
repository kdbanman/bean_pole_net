mkdir backup
mkdir backup/log
mkdir backup/checkpoints

for run in "$@"
do
    echo "backing up $run"
    cp -r log/$run backup/log/
    cp -r checkpoints/$run backup/checkpoints
done

echo "now remove those runs (if you want), rename the directory, add a readme, and tar it up"
