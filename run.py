"""
"""

import schedule
import time
import ip_camera_control
import evaluate

#Mode options = 'training' or 'evaluate'
# Training saves images to the 'traning/uncategorized' photo for manual sorting and later training
# Evaluate captures photos and predicts their class based on a pre-trained model which you can create with train_model.ipynb

MODE = 'evaluate'
TRAINING_IMG_PATH = 'training/uncategorized'
EVALUATE_IMG_PATH = 'eval_img'
PROJECT_NAME = 'kc' # Used as prefix on all photos to distinguish projects
CAPTURE_FREQ = 1 # how frequently the camera takes snaps. In minutes
RUN_DURATION = 20 # amount of time to run before stopping, in minutes

def evaluate_func(capture):
    print(time.strftime("%Y-%m-%d %H:%M", time.localtime()))
    evaluate.main(capture)
    return

def training_func(capture):
    print(time.strftime("%Y-%m-%d %H:%M", time.localtime()))
    filepath = capture.get_snap()
    return

def run_scheduler(func_to_run, capture, capture_freq, run_duration):

    start_time = time.time()
    time_since = 0
    func_to_run(capture) #run the first time
    schedule.every(capture_freq).minutes.do(func_to_run, capture) #schedule subsequent times

    while time_since < run_duration*60.0:
        schedule.run_pending()
        time.sleep(1)
        time_since = time.time() - start_time

    return


def run(MODE, TRAINING_IMG_PATH, EVALUATE_IMG_PATH, PROJECT_NAME, CAPTURE_FREQ, RUN_DURATION):

    assert (MODE in ['training', 'evaluate'])

    if MODE == 'evaluate':
        output_path = EVALUATE_IMG_PATH
        func = evaluate_func
    elif MODE == 'training':
        output_path = TRAINING_IMG_PATH
        func = training_func
    else:
        print('Mode error')
        return

    capture = ip_camera_control.Capture(output_path = output_path, project_name = PROJECT_NAME)
    run_scheduler(func,capture,CAPTURE_FREQ,RUN_DURATION)

    print('finished')

    return

if __name__ == "__main__":
    run(MODE, TRAINING_IMG_PATH, EVALUATE_IMG_PATH, PROJECT_NAME, CAPTURE_FREQ, RUN_DURATION)
