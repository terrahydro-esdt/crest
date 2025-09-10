from multiprocessing import Process, Lock
from flask import Flask, jsonify
from crest.utils.crest_logger import logger_setup, install_global_exception_logger
from contextlib import contextmanager
from datetime import datetime
import time
import pandas as pd
import logging
import os

metrics_df = pd.DataFrame(columns=["timestamp", "operation", "duration_sec", "status", "detail"])

logger_setup(log_file="crest.log.jsonl", metrics_file="metrics.log.jsonl")

install_global_exception_logger()

last_saved_index = -1  # Global tracker

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.logger.propagate = True

dre = [None, None, None, None]
dre_count = 0
nowcast_lock = Lock()
is_nowcasting = False  # shared flag

def run_nowcast(index):
    from crest.engine.DigitalReplicaEngine import DigitalReplicaEngine

    model_loader = None

    # if terrahydro is installed, use its model loader
    if ("terrahydro" in globals() and 
        hasattr(terrahydro, "models") and 
        hasattr(terrahydro.models, "utils") and 
        hasattr(terrahydro.models.utils, "ModelServer") and 
        hasattr(terrahydro.models.utils.ModelServer, "ModelLoader")
    ):
        model_loader = terrahydro.models.utils.ModelServer.ModelLoader
        import data.mh, data.coupled, data.uncoupled

    logger.info(f"Starting nowcast run with index {index}")

    try:
        if index == 0:
            logger.debug("Using mh.yaml configuration")
            local_dre = DigitalReplicaEngine('data/mh.yaml', model_loader=model_loader, process_dataset=data.mh.process_dataset)
        elif index == 1:
            logger.debug("Using coupled.yaml configuration")
            local_dre = DigitalReplicaEngine('data/coupled.yaml', model_loader=model_loader, process_dataset=data.coupled.process_dataset)
        elif index == 2:
            logger.debug("Using uncoupled.yaml configuration")
            local_dre = DigitalReplicaEngine('data/uncoupled.yaml', model_loader=model_loader, process_dataset=data.uncoupled.process_dataset)
        elif index == 3:
            logger.debug("Using config.yaml configuration")
            local_dre = DigitalReplicaEngine('data/config.yaml')
        else:
            logger.error(f"Invalid DRE index passed: {index}")
            raise ValueError(f"Invalid dre index: {index}")

        with nowcast_lock:
            logger.info(f"Acquired lock. Running nowcast for index {index}")
            local_dre.nowcast()
    except Exception as e:
        logger.exception(f"Exception during nowcast execution for index {index}: {e}")

@app.route('/run-dre/<int:dre_index>')
def start_workers(dre_index):
    global nowcast_lock

    logger.info(f"Received request to run DRE index {dre_index}")

    if not (0 <= dre_index < len(dre)):
        logger.warning(f"Invalid DRE index {dre_index}")
        return jsonify(status="error", message="Invalid DRE index"), 400

    if nowcast_lock.acquire(block=False):
        try:
            logger.info(f"Starting new process for DRE index {dre_index}")
            p = Process(target=run_nowcast, args=(dre_index,))
            p.start()
            return jsonify(status="success", message="Nowcast started"), 200
        finally:
            nowcast_lock.release()
            logger.debug("Released nowcast lock after spawning process")
    else:
        logger.warning("Nowcast already running. Rejecting request.")
        return jsonify(status="error", message="Nowcast already running"), 429

@app.route('/metrics')
def get_metrics():
    global last_saved_index
    try:
        os.makedirs("outputs", exist_ok=True)
        csv_path = os.path.join("outputs", "operations.csv")
        file_exists = os.path.isfile(csv_path)

        new_rows = metrics_df.iloc[last_saved_index + 1:]
        if not new_rows.empty:
            new_rows.to_csv(
                csv_path,
                mode='a',
                header=not file_exists,
                index=False
            )
            logger.debug(f"Appended {len(new_rows)} new row(s) to {csv_path}")
            last_saved_index = new_rows.index[-1]  # Update tracker
    except Exception as e:
        logger.exception(f"Failed to append metrics to CSV: {e}")

    return jsonify(metrics_df.tail(20).to_dict(orient='records'))

    
@app.route('/list-dres')
def list_dres():
    logger.info("Listing DRE statuses")
    status = ["initialized" if dre[i] else "not initialized" for i in range(len(dre))]
    return jsonify({f"DRE {i}": s for i, s in enumerate(status)})

@app.route('/health')
def health():
    logger.debug("Health check received")
    return "OK", 200

if __name__ == '__main__':
    from multiprocessing import set_start_method
    set_start_method("spawn", force=True)  # Ensures safe multiprocessing

    logger.info("Starting Flask app for Nowcast Runner")
    app.run(host='0.0.0.0', port=5000, debug=False)
