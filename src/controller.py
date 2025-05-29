import numpy as np


class ControllerDressing:
    def __init__(self):
        self.start_dressing = False
        self.dressing_wrist_finished = False
        self.dressing_success = False
        self.dress_wrist = False
        self.dress_shoulder = False


    def get_actions(self, x=0, y=0, dress=False):
        if dress:
            action = x - y
            err = np.linalg.norm(action)
            action /= err
        else:
            action = np.zeros(3)
            err = 100

        if err < 0.05:
            action = np.zeros(3)

        return action, err
    
    def meta_action(self, arm_pos, ee_pos):
        wrist_3d = arm_pos[2]
        wrist_ee_l = np.linalg.norm(wrist_3d)
        if (wrist_ee_l < 0.3) and (not self.dress_wrist):
            self.dress_wrist = True
        else:
            self.dress_wrist = False
            action, err = self.get_actions()
        
        if self.dress_wrist and not self.dressing_wrist_finished:
            action, err = self.get_actions(arm_pos[2], ee_pos, dress=True)
            if err < 0.05:
                self.dressing_wrist_finished = True
            

        if self.dressing_wrist_finished and self.dress_wrist:
            action, err = self.get_actions(arm_pos[0], ee_pos, dress=True)
        
        return action, err
