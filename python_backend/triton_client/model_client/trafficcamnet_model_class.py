import os
import requests
from base_model_class import BaseModelClass
from trafficcamnet_client import trafficcamnet_predict
from requests.exceptions import ConnectionError


class TrafficCamNetModelClass(BaseModelClass):

    def __init__(self, client_info):
        '''
        Instantiate the classes with the information of the 
        querying party -- corresponding to a specific triton
        model.
        '''
        BaseModelClass.__init__(self, client_info)
        self._post_processing_config = "/app/triton_client/model_client/tao_triton/python/clustering_specs/clustering_config_trafficcamnet.prototxt"
        self._url = os.environ.get('API_URL')
        self._model_name = "trafficcamnet"
        self._mode = "trafficcamnet"
        self._class_list = "car,bicycle,person,road_sign"

    def status(self):
        '''
        Returns the status of the model
        '''
        try:
            triton_server_url = "http://" + self._url + "/v2/health/ready"
            response = requests.get(triton_server_url)
        except ConnectionError as error:
            return {'HTTPStatus': 503, 'status': 'Inactive'}
        else:
            return {'HTTPStatus': 200, 'status': 'Active'}

    def predict(self, file_path):
        '''
        Returns the output of the model after inference.
        '''
        if os.path.exists(file_path):
            return self._predict(file_path)
        else:
            return [{'HTTPStatus': 400,
                     'error': "File Path does not exist!"}]

    def _predict(self, file_path):
        number_files = len([name for name in os.listdir(
            file_path) if os.path.isfile(file_path+name)])
        if number_files < 256:
            self._batch_size = 8
        else:
            self._batch_size = 16
        return trafficcamnet_predict(model_name=self._model_name, mode=self._mode, class_list=self._class_list,
                           output_path="./", postprocessing_config=self._post_processing_config,
                           url=self._url, image_filename=file_path, verbose=False, streaming=False, async_set=False,
                           protocol='HTTP', model_version="", batch_size=self._batch_size)

import cv2 as cv
def render_box(results, img_dir, out_dir):
    for res in results:
        file_name = res['file_name']
        img = cv.imread(f'{img_dir}/{file_name}')
        for b in res['all_bboxes']:
            b = [int(coord) for coord in b['bbox']]
            x1, y1, x2, y2 = b
            img = cv.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)  
        cv.imwrite(f'{out_dir}/{file_name}', img)



# To handle output_path
if __name__ == "__main__":
    test_model = TrafficCamNetModelClass("hellosss")
    # print(test_model.status())
    # res = test_model.predict("../input/lpd")

    results_match = test_model.predict('../sample_images/match')
    render_box(results_match, '../sample_images/match', '../experiment/tc/4/match')

    results_over = test_model.predict('../sample_images/over')
    render_box(results_over, '../sample_images/over', '../experiment/tc/4/over')


    results_miss = test_model.predict('../sample_images/miss')
    render_box(results_miss, '../sample_images/miss', '../experiment/tc/4/miss')
