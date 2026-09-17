from tensorflow.keras.layers import LSTM, Bidirectional

def create_bidirectional_lstm():
    lstm_fw = LSTM(units=32)
    lstm_bw = LSTM(units=32, go_backwards=True)
    return Bidirectional(lstm_fw, backward_layer=lstm_bw)
