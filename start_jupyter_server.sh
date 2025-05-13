ipnport=$(shuf -i 8000-8500 -n 1)
export JUPYTER_RUNTIME_DIR=/tmp/jupyter_runtime
mkdir -p $JUPYTER_RUNTIME_DIR
jupyter notebook --no-browser --port=${ipnport} --ip=$(hostname -i)
