from xrd_assistant import *


reference_phases = sorted(os.listdir('References/'))
model = tf.keras.models.load_model('Model.h5', custom_objects={'sigmoid_cross_entropy_with_logits_v2': tf.nn.sigmoid_cross_entropy_with_logits})
kdp = KerasDropoutPrediction(model)

for fname in os.listdir('Spectra/'):
    total_confidence, all_predictions = [], []
    tabulate_conf, predicted_cmpd_set = [], []
    mixtures, confidence = explore_mixtures('Spectra/%s' % fname, kdp, reference_phases)
    max_conf_ind = np.argmax(confidence)
    max_conf = 100*confidence[max_conf_ind]
    predicted_cmpds = mixtures[max_conf_ind]
    if len(predicted_cmpds) == 1:
        predicted_set = '%s' % predicted_cmpds[0][:-4]
    if len(predicted_cmpds) == 2:
        predicted_set = '%s + %s' % (predicted_cmpds[0][:-4], predicted_cmpds[1][:-4])
    if len(predicted_cmpds) == 3:
        predicted_set = '%s + %s + %s' % (predicted_cmpds[0][:-4], predicted_cmpds[1][:-4], predicted_cmpds[2][:-4])
    print('Filename: %s' % fname)
    print('Predicted phases: %s' % predicted_set)
    print('Associated confidence: %s\n' % max_conf)
