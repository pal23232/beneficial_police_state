""""
This module takes an image and evaluates it against an existing ML model to classify it against pre-defined classes.

It then logs the results in a logfile

There is an additional logic here to post the results to Slack based on certain criteria. You will probably want to change the logic and reporting behavior based on your own application.

The example used here is evaluating whether a kitchen in a house is "clean" "dirty" or has people in it ("peoople"). If the kitchen is dirty for 3 pictures in a row and changed from another state, it will post to Slack.


"""

from PIL import Image
import torch
import torchvision
import os
import time
import requests
import csv
import shutil
import requests
import random
import config
import sys

## Import fastai library. Included in Git repository
FASTAI_PATH = 'fastai'
sys.path.append(FASTAI_PATH)
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

# PARAMETERS
ML_MODEL_PATH = 'models/180322-resize'  ##Saved torch model
CLASSES = ['clean', 'dirty', 'people'] ###Enter the clases for your application here
LOG_FILEPATH = 'log/log.csv' # log file for evaluations
IMG_STORE_DIR = 'img_store' # where to store categorized images
IMAGE_STORE_MODE = 2 # 0 = leave image in folder. 1 = delete image. 2 = move to image_store_dir sorted by category
NUM_SAME = 3 #For the posting criteria. Number of consecutive images that need to be the same for it to post to Slack.
ALWAYS_POST = False ##Testing setting. True means it will post every picture to Slack. Usually False.
RESIZE_IMAGES = True ## Resize images to a set sized square to fit the ML model.
RESIZE_SIZE = 224 #doesn't matter if resize_images = false

def main(capture):

    filepath = capture.get_snap()
    time.sleep(5)

    valid_file = check_valid_file(filepath)
    print(filepath, valid_file)

    if valid_file:

        model = Model(ML_MODEL_PATH, CLASSES)
        img_input_to_model = model.prepare_image(filepath, RESIZE_IMAGES, RESIZE_SIZE)
        pred_class, pred_prob, categorize_success = model.categorize(img_input_to_model)

        print(pred_class, pred_prob, categorize_success)


    if categorize_success is False or valid_file is False:
        pred_class = 'uncategorized'
        pred_prob = ''

    log_results(LOG_FILEPATH, filepath, valid_file, pred_class, pred_prob)

    if categorize_success and valid_file:
        slack = SlackPoster()
        post = slack.decide_to_post(LOG_FILEPATH, NUM_SAME)

        if post or ALWAYS_POST:
            slack.post_to_slack(filepath, pred_class)

    if IMAGE_STORE_MODE == 2:
        move_photo(filepath,pred_class, IMG_STORE_DIR)
    elif IMAGE_STORE_MODE == 1:
        os.remove(filepath)

    return

def check_valid_file(filepath):

    TIME_WINDOW_SECS = 180
    MIN_SIZE_BYTES = 20000
    valid_file = True

    if os.path.getsize(filepath) < MIN_SIZE_BYTES:
        valid_file = False

    if time.time() - os.path.getctime(filepath) > TIME_WINDOW_SECS:
        valid_file = False

    return valid_file


class Model(object):

    def __init__(self, ml_model_path, classes):
        self.model = torch.load(ml_model_path, map_location=lambda storage, loc: storage)
        self.classes = classes

    def prepare_image(self, filepath, resize_images = False, sz=224):
        self.filepath = filepath
        img = Image.open(filepath)

        if resize_images:
            img = img.resize((sz,sz), resample = PIL.Image.LANCZOS)

        img_variable = self.preprocess_image(img)

        return img_variable

    def preprocess_image(self,img):
        #applies to Resnet models
        normalize = torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
        preprocess = torchvision.transforms.Compose([ torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(), normalize ])
        img_tensor = preprocess(img).unsqueeze_(0)
        return Variable(img_tensor)

    def categorize(self, img_input_to_model):

        pred_prob = 0.0
        pred_class = ''
        categorize_success = False
        try:
            log_probs = self.model(img_input_to_model)
            probs = np.exp(log_probs.data.numpy())
            pred = int(np.argmax(log_probs.cpu().data.numpy(), axis=1))
            pred_class = self.classes[pred]
            pred_prob = probs[0,pred]
            categorize_success = True
        except:
            print('categorization failed')

        return pred_class, pred_prob, categorize_success


def log_results(LOG_FILEPATH, filepath, valid_file, pred_class, pred_prob):

    file_exists = os.path.isfile(LOG_FILEPATH)

    with open(LOG_FILEPATH,'a') as fd:
        #writer = csv.writer(fd)
        current_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        header_row = ['timestamp', 'filename', 'valid_file', 'pred_class', 'pred_prob']
        writer = csv.DictWriter(fd, delimiter=',', lineterminator='\n',fieldnames=header_row)
        myCsvRow = {'timestamp':current_time, 'filename':os.path.basename(filepath), 'valid_file':valid_file, 'pred_class':pred_class, 'pred_prob':pred_prob}

        if not file_exists:
            writer.writeheader()

        writer.writerow(myCsvRow)

    return

def move_photo(filepath,pred_class, IMG_STORE_DIR):


    move_dir = os.path.join(IMG_STORE_DIR, pred_class)

    try:
        if not os.path.exists(move_dir):
            os.makedirs(move_dir)
        shutil.move(filepath, move_dir)
    except:
        print('move to %s failed') % move_dir
    return

def photo_cleanup():
    # Could add a function to cleanup old photos
    return


class SlackPoster(object):

    def __init__(self):
        self.token = config.SLACK_TOKEN_BOT
        self.channel = config.SLACK_CHANNEL

    def post_to_slack(self,filepath, pred_class):

        MESSAGE = pred_class
        url = 'https://slack.com/api/files.upload'

        payload = {
            'token':self.token,
            'channels':self.channel,
            'title':MESSAGE,
            'filename':os.path.basename(filepath)
        }

        files = {'file' : (filepath, open(filepath, 'rb'), 'image/png')}

        try:
            r = requests.post(url, params=payload, files=files)

        except requests.exceptions.RequestException as e:
            print("Error: {}".format(e))

        return

    def decide_to_post(self,LOG_FILEPATH, NUM_SAME):
        post = True

        with open(LOG_FILEPATH, 'r') as f:
            entries = list(csv.DictReader(f))

        if len(entries) < NUM_SAME + 1:
            print('take more photos first')
            return False

        current_entries = entries[-1*NUM_SAME:]
        prior_entry = entries[-1*NUM_SAME-1]
        current_class = current_entries[NUM_SAME-1]['pred_class']
        prior_class = prior_entry['pred_class']
        classes = [entry['pred_class'] for entry in current_entries]
        valid_files = [entry['valid_file'] for entry in current_entries]

        # CHECK IF CURRENT ENTRIES

        #CHECK IF THERE'S BEEN A CHANGE
        if current_class == prior_class:
            print('no class change')
            post = False

        #check to see if last ones have all been valid
        if valid_files.count('False') > 0 or classes.count('uncategorized') > 0:
            print('not all valid')
            post = False

        #check to see if consistent across NUM_SAME
        if classes.count(current_class) < NUM_SAME:
            print('recents not all same class')
            post = False

        #check to see if pred_class is dirty
        if current_class != 'dirty':
            print('not dirty')
            post = False

        if post:
            print('Posting to Slack')

        return post
