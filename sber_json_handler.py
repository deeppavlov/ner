import json
from dilated_ner import DilatedNER
import os
from dilated_ner import MODEL_PATH
from dilated_ner import MODEL_FILE_NAME
from corpus import Corpus
from corpus import data_reader_gareev


REQUEST_FILE_NAME = 'sbrf.json'
RESPONSE_FILE_NAME = 'response.json'


def predict_for_json_request(json_filename):
    with open(json_filename) as f:
        requests = json.load(f)
    session_id = requests['id']
    assert requests['type'] == 'kpi3'
    answers = []
    for request in requests['qas']:
        id_ = request['id']
        question = request['question']
        x, xc = corp.tokens_to_x_xc(question.split())
        result = ' '.join(ner_model.predict(x, xc)[0])
        answers.append({id_: result})
    response_dict = {'sessionId': session_id, 'answers': answers}
    return response_dict


if __name__ == '__main__':
    corp = Corpus(data_reader_gareev)

    n_layers_per_block = 3
    n_blocks = 1
    dilated_filter_width = 5
    embeddings_dropout = True
    dense_dropout = True
    model_path = os.path.join(MODEL_PATH, MODEL_FILE_NAME)
    print('L')
    ner_model = DilatedNER(corp,
                           n_layers_per_block=n_layers_per_block,
                           n_blocks=n_blocks,
                           dilated_filter_width=dilated_filter_width,
                           embeddings_dropout=embeddings_dropout,
                           dense_dropout=dense_dropout,
                           pretrained_model_filepath=model_path)
    resp_dict = predict_for_json_request(REQUEST_FILE_NAME)
    with open(RESPONSE_FILE_NAME, 'w') as f:
        json.dump(resp_dict, f)
