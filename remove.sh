for run in "$@"
do
    echo "removing $run"
    rm -rf log/$run
    rm -rf checkpoints/$run
done