import json
from loader.kitti_road_loader import KITTIRoadLoader

def get_loader(name):
    return {
        'kitti_road': KITTIRoadLoader,
    }[name]

def get_data_path(name, config_file='config.json'):
    data = json.load(open(config_file))
    return data[name]['data_path']
    