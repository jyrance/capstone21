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
                           url=self._url, image_filename=file_path, verbose=False, streaming=False, async_set=True,
                           protocol='HTTP', model_version="", batch_size=self._batch_size)

# To handle output_path
if __name__ == "__main__":
    test_model = TrafficCamNetModelClass("hellosss")
    # print(test_model.status())
    res = test_model.predict("../input/lpd")

    import pickle
    with open('output.pickle', 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)