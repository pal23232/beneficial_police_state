import requests
import shutil
import config
import time
import os
import sched

class Capture(object):

    def __init__(self, output_path = 'snaps/', project_name = 'kc'):
        self.cam_url, self.username, self.pw = self.get_cam_config()
        self.project_name = project_name
        self.output_path = output_path
        return

    def get_cam_config(self):
        username = config.USERNAME
        pw = config.PW
        cam_url = config.CAM_URL
        return cam_url, username, pw


    def set_image_name(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        return self.output_path+self.project_name+'-'+time.strftime("%Y%m%d-%H%M%S")+'.png'

    def get_snap(self):
        snap_url = '%s/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=wuuPhkmUCeI9WG7C&user=%s&password=%s' % (self.cam_url, self.username, self.pw)

        r = requests.get(snap_url, stream=True)
        if r.status_code == 200:

            with open(self.set_image_name(), 'wb') as out_file:
                shutil.copyfileobj(r.raw, out_file)
            del r
        else:
            print('failed with status code:'+r.status_code)

        return

    def regular_capture(self, capture_freq = 1.0, num_captures  = 400):
        #capture frequency in minutes
        s = sched.scheduler(time.time, time.sleep)
        for i in range(num_captures):
            s.enter((i+1)*capture_freq*60, 1, self.get_snap, ())

        s.run()
        return

    ##TODO Create function to clean old snaps

    # def clean_old_snaps():
    #     for f in os.listdir(path):
    #         if os.stat(os.path.join(path,f)).st_mtime < now - 7 * 86400

def main():
    capture = Capture()
    capture.regular_capture(capture_freq = 0.1, num_captures  = 5)
    return

if __name__ == "__main__":
    main()
