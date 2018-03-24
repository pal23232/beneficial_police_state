## Controls the IP camera

import requests
import shutil
import config
import time
import os

class Capture(object):

    def __init__(self, output_path = 'training/uncategorized', project_name = 'kc'):

        self.cam_url, self.username, self.pw = self.get_cam_config()
        #URL used to access the IP camera over the network. This applies to the Reolink P1 Pro.
        # https://reolink.com/wp-content/uploads/2017/01/Reolink-CGI-command-v1.61.pdf
        self.SNAP_URL = '%s/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=wuuPhkmUCeI9WG7C&user=%s&password=%s' % (self.cam_url, self.username, self.pw)
        self.project_name = project_name
        self.output_path = output_path
        return

    def get_cam_config(self):
        # you will need to set up a config file with the login info for your camera
        username = config.USERNAME
        pw = config.PW
        cam_url = config.CAM_URL
        return cam_url, username, pw


    def set_image_name(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        return os.path.join(self.output_path,self.project_name+'-'+time.strftime("%Y%m%d-%H%M%S")+'.png')

    def get_snap(self):

        filepath = self.set_image_name()
        r = requests.get(self.SNAP_URL, stream=True)
        if r.status_code == 200:
            with open(filepath, 'wb') as out_file:
                shutil.copyfileobj(r.raw, out_file)
            del r
        else:
            print('failed with status code:'+r.status_code)

        return filepath
