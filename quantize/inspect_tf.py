# import debugpy

# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for client to attach...")
# debugpy.wait_for_client()
import tensorflow as tf
model = tf.keras.models.load_model('models/CNN/weights.100-23.14.hdf5',compile=False)
model.compile()
model.summary()
from tensorflow_model_optimization.quantization.keras import vitis_inspect
inspector = vitis_inspect.VitisInspector(target="0x101000016010407")
inspector.inspect_model(model, 
                        plot=True, 
                        plot_file="model.svg", 
                        dump_results=True, 
                        dump_results_file="inspect_results.txt", 
                        verbose=0,
                         )