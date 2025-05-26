import numpy as np
from watermark_operations import embed_ml_assisted, extract_ml_assisted
from ml_training.data_loader import load_predictor_dataset

def test_roundtrip():
    # dummy cover
    cover = (np.random.rand(256,256)*255).astype('uint8')
    # dummy bits
    bits = np.random.randint(0,2,size=(1000,)).astype('uint8')

    # load models
    from tensorflow.keras.models import load_model
    import joblib
    predictor = load_model('models/ml_assisted/predictor.keras')
    thres_reg = joblib.load('models/ml_assisted/thres_regressor.pkl')
    region_clf = load_model('models/ml_assisted/region_classifier.keras')

    wm = embed_ml_assisted(cover, bits, predictor, thres_reg, region_clf)
    rec_bits, orig = extract_ml_assisted(wm, predictor, thres_reg, region_clf)

    assert np.array_equal(orig, cover), "Original not recovered exactly!"
    assert np.array_equal(rec_bits[:len(bits)], bits), "Watermark bits mismatch!"
    print("ML-Assisted roundtrip test passed.")

if __name__=='__main__':
    test_roundtrip()
