# Image classification via IP security camera

My first Github contribution, so pls be kind.

## What is this?

Getting us one small step closer to a beneficial police state.

This code helps you use a Wifi-enabled IP Camera to do scene classification.

Example applications:
+ Determine whether your kitchen is clean, dirty, or in-use. Send notice to housemates when it needs to be cleaned (this is the application shown in the current code)
+ See what % of the time a room is occupied with people
+ .. I dunno .. Record what time the mailman arrives everyday?

## What do I need?
+ Python 3.6.4+
+ Pytorch 0.3.1+, FastAI library (included in Repo), Torchvision
+ Jupyter notebook to step through the model training code
+ An IP Camera. We use the Reolink P1 Pro, which works well. Make sure it has an API that enables you to control it via an HTTP request (i.e. Nest will not work)
+ Numpy, PIL, requests, schedule,
+ Optional: A GPU to train the model, but can also work on CPU, just slower. We use Paperspace

## How do I use it?

Step 1: Install IP Camera and figure out how to control it in ip_camera_control.py.  Configuration for Reolink cams included, but will need to be modified for others.

Step 2: Decide what you want to classify. We did "clean" vs "dirty" vs "people" (aka in use).

Step 3: Set "MODE" to "training" in run.py. Collect a bunch of photos. We collected ~300 images of our kitchen over the course of a week for training.It's helpful to set the "capture_freq" to a time when you expect the scene to change, so you aren't classifying repeat images.

Step 4: Manually categorize the images into classes

Step 5: Train the model using train_model.ipynb. It will create the proper folder structure. Save the model. We used a pre-trained ResNet34 model and finetuned it, which resulted in ~85% accuracy.

Step 6: Now you can classify. Adjust the parameters in "evaluate.py." Set "MODE" to "evaluate" in "run.py" and go ahead and run the program. It will classify images based on the model, and log + file them appropriately.

Step 7 (optional + customizable): Take some action based on the classification. Right now it's programed to post to a Slack channel when the kitchen is dirty for at least 21 minutes consecutively. This will vary by application. 
