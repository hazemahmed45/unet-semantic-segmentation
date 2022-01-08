import os
import json


class Configs():
    def __init__(self) -> None:
        self.experiment_num=15
        self.exp_dir='exp'
        if(not os.path.exists(self.exp_dir)):
            os.mkdir(self.exp_dir)
        self.exp_num_file=os.path.join(self.exp_dir,'exp_num.txt')
        if(os.path.exists(self.exp_num_file)):
            with open(self.exp_num_file,'r') as f:
                self.experiment_num=int(f.readline().replace("\n",""))
            self.experiment_num+=1
        with open(self.exp_num_file,'w') as f:
            f.write(str(self.experiment_num))
        
        self.cur_exp_dir=os.path.join(self.exp_dir,str(self.experiment_num))
        if(not os.path.exists( self.cur_exp_dir)):
            os.mkdir( self.cur_exp_dir)
        self.is_stoped=False
        self.epochs=10
        self.pin_memory=True
        self.random_seed=999
        self.low_aug_bounds=0.5
        self.high_aug_bounds=0.7
        self.img_dir='Dataset/train/images'
        self.mask_dir='Dataset/train/masks'
        self.img_width=101
        self.img_height=101
        self.batch_size=64
        self.shuffle=True
        self.augmentation=True
        self.num_workers=8
        self.lr=1e-3
        self.train_split=0.8
        self.loss_magnifier=1
        self.schedular='step'
        self.step_size=10
        self.step_gamma=0.45
        self.job_type='segmentation'
    def get_config_dict(self):
        return vars(self)
    def save_config_dict(self):
        with open(os.path.join(self.cur_exp_dir,'config.json'),'w') as f:
            json.dump(self.get_config_dict(),f)
        return 
    def load_config_dict(self,config_file):
        config_dict={}
        with open(config_file,'r') as f:
            config_dict=json.load(f)
        for key,value in config_dict.items():
            self.__setattr__(key,value)
            
            
if (__name__ == '__main__'):
    config=Configs()
    print(config.get_config_dict())
    config.save_config_dict()