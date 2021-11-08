import tensorflow as tf

def get_model():
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.InputLayer(input_shape=[40, ]))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.3))
	model.add(tf.keras.layers.Dense(256, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.3))
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dense(10, activation='softmax'))
	model.compile(metrics = 'accuracy', optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy')

	return model

def train_model(X_train, y_train, X_test, y_test, epochs, save_at):
	model = get_model()
	checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath = save_at + '/Sound_Classifier.h5', verbose=1, save_best_only=True)
	model.fit(x=X_train, y = y_train, epochs = 100, validation_data=(X_test, y_test), callbacks=[checkpoint_cb], verbose = 1)