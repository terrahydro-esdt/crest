from flask import Flask
from multiprocessing import Process
from crest.engine.DigitalReplicaEngine import DigitalReplicaEngine
import logging
from crest.utils.setup_logging import logger_setup, install_global_exception_logger

logger_setup("dev")  # or "prod", etc.
install_global_exception_logger()
logging.getLogger('werkzeug').setLevel(logging.INFO)

# Initialize the app
app = Flask(__name__)

def init_dre():
    global dre 
    dre = DigitalReplicaEngine('data/config.yaml')

@app.route('/run-dre')
def start_workers():
    global dre
    
    dre.nowcast()

    return 'Nowcast started!'

# Run the app
if __name__ == '__main__':
    with app.app_context():
        init_dre()
        
    app.run(debug=True)
