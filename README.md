## How to install TF Serving for Keras models

These are the notes that could help you to export your trained Keras
model(s), run Tensorflow Serving gRPC server to provide an API
and then have a Python scripts making calls to it over gRPC.

The Python script can be Flask that does data preprocessing while
TFS provides very low level API to the NN model.

All this can be done in 3 difficult steps:
- export Keras model for use in TFS
- compile and run TFS to serve the model
- create and compile Python script that can "talk" gRPC with TFS


Note: TFS only works with Python2


### How to export your Keras model to use with TFS

Code examples:
https://github.com/krystianity/keras-serving/blob/master/export.py
https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html#exporting-a-model-with-tensorflow-serving

I'm using a script that scans a tree if directories and runs subprocess to export the model.
Be careful, you can't export multiple models in a single Python process. Each additional
model adds more variables to computational graph and I'm not aware how it can be cleaned,
hence my subprocess thing.

Example snippet:

```python
# Don't import tf here, your Python script shouldn't know what tf/keras is, all this is contained within a subprocess

def export_tfs(dst_dir, model_name):
    """
    dst_dir - directory name where Keras model is located
    
    There should be two files:
    {model_name}.json - model structure
    {model_name}.h5f - weights

    Exported model will be saved into dst_dir/tfserving/1/
    """
    from tensorflow.python.saved_model import builder as saved_model_builder
    from tensorflow.python.saved_model import tag_constants, signature_constants
    from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
    from keras import backend as K
    from keras.models import model_from_json

    with K.get_session() as sess:
        K.set_learning_phase(0)

        with open(os.path.join(dst_dir, model_name + '.json')) as f:
            model = model_from_json(f.read())
        model.load_weights(os.path.join(dst_dir, model_name + '.h5f'))
        model.summary()

        builder = saved_model_builder.SavedModelBuilder(os.path.join(dst_dir, 'tfserving/1'))
        signature = predict_signature_def(inputs={"inputs": model.input},
                                          outputs={"outputs": model.output})
        builder.add_meta_graph_and_variables(sess=sess,
                                 tags=[tag_constants.SERVING],
                                 signature_def_map={'predict': signature})
        builder.save()

...

# Exporting for TF Serving
p = multiprocessing.Process(target=export_tfs, args=(dst_dir, model_name))
p.start()
print('Exporting {} in process {} ...'.format(dst_dir, p.pid))
p.join()
```

I assume you do have a few exported models in directories:
~/models/model1/tfserving/1/
~/models/model2/tfserving/1/

The main point here is that directory X should have subdirectory '1' with the actual model.
You should provide path to X in TFS config file later.


### How to build and run TFS

1) Install Bazel. Bazel is a build system that is used in TF and TFS

https://docs.bazel.build/versions/master/install-ubuntu.html

echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel

2) Follow the guide to install all prerequisites for TFS, but don't clone TFS just yet

https://tensorflow.github.io/serving/setup

3) I assume that TFS client will be located in ~/client directory

```sh
cd
mkdir client
cd client
git clone --recurse-submodules https://github.com/tensorflow/serving
cd serving
```

At this point "serving" has TFS source tree inside of "~/client" directory
TFS has copy of TF tree in serving/tensorflow

4) Configure TF

```sh
cd tensorflow
./configure
cd ..
```

When configuring accept answers by default, make sure to stick to Python2

5) Build TFS gRPC server

```sh
bazel build tensorflow_serving/...
bazel test tensorflow_serving/...
```

6) Create TFS config file

There are examples on internet that don't use config file and specify path to model in cmdline, but config file
allows to serve multiple models with a single TFS server
Credit: https://stackoverflow.com/a/43745791

Example of tfserving.conf file:

```
model_config_list: {
  config: {
    name: "model1",
    base_path: "models/model1/tfserving/1",
    model_platform: "tensorflow"
  },
  config: {
     name: "model2",
     base_path: "models/model2/tfserving/1",
     model_platform: "tensorflow"
  }
}
```

7) Run TFS:

```sh
~/client/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_config_file=tfserving.conf
```

### How to build a client script that would interact with running TFS

That's the most difficult part.
Directory structure reminder:
~/client/serving - that's where TFS is

1) Go to ~/client and create following files there:

~/client/WORKSPACE

```
workspace(name = "my_project")

local_repository(
    name = "tf_serving",
    path = __workspace_dir__ + "/serving/",
)

local_repository(
    name = "org_tensorflow",
    path = __workspace_dir__ + "/serving/tensorflow/",
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "4be8a887f6f38f883236e77bb25c2da10d506f2bf1a8e5d785c0f35574c74ca4",
    strip_prefix = "rules_closure-aac19edc557aec9b603cd7ffe359401264ceff0d",
    urls = [
        "http://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/aac19edc557aec9b603cd7ffe359401264ceff0d.tar.gz",  # 2017-05-10
        "https://github.com/bazelbuild/rules_closure/archive/aac19edc557aec9b603cd7ffe359401264ceff0d.tar.gz",
    ],
)

load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')
tf_workspace("serving/tensorflow/", "org_tensorflow")
```


~/client/BUILD

```
py_binary(
    name = "my_api_gw",
    srcs = [
        "my_api_gw.py",
    ],
    deps = [
        "@tf_serving//tensorflow_serving/apis:predict_proto_py_pb2",
        "@tf_serving//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
```

~/client/my_api_gw.py

```python

from __future__ import print_function

import numpy as np
import tensorflow as tf

from google.protobuf.json_format import MessageToJson
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

# Assuming model's input is 10 float numbers and there are 3 samples in the batch
data = [[0.42] * 10] * 3

def main():
    channel = implementations.insecure_channel('localhost', 9000)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'model2'   # model name as in TFS config file
    request.model_spec.signature_name = 'predict'  # this should always be 'predict'

    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(data, shape=[3, 10]))

    result = stub.Predict(request, 5.0)  # 5 seconds
    preds = np.array(result.outputs['outputs'].float_val).reshape((result.outputs['outputs'].tensor_shape.dim[0].size,
                                                                   result.outputs['outputs'].tensor_shape.dim[1].size))
    print(preds.shape)


if __name__ == '__main__':
  main()
```

This script will send a batch of 3 samples (10 floats in each) to model "model2" and print array shape of received result

3) Build your script

```sh
cd ~/client
bazel build my_api_gw    # as in BUILD file
```

4) Patch one of the files in your build

https://github.com/tensorflow/serving/issues/421

cd ~/client
vim bazel-bin/my_api_gw.runfiles/org_tensorflow/tensorflow/contrib/image/__init__.py

Go to imports section and comment out following line:

```python
from tensorflow.contrib.image.python.ops.single_image_random_dot_stereograms import single_image_random_dot_stereograms
```

5) Run the script
~/client/bazel-bin/my_api_gw

```sh
user@pc ~/client % bazel-bin/my_api_gw
(3, 103)
```

### Known problems:
- Without fix for TFS ticket 421 your client script won't run complaining about missing library:
  ```
  tensorflow.python.framework.errors_impl.NotFoundError: /home/usr/serving/bazel-bin/tensorflow_serving/example/mnist_export.runfiles/org_tensorflow/tensorflow/contrib/image/python/ops/_single_image_random_dot_stereograms.so: undefined symbol: _ZN6google8protobuf8internal10LogMessageC1ENS0_8LogLevelEPKci
  ```

- TFS source tree has subpackage "tensorflow". When you run "./configure" you do configure just "tensorflow"
  You could build TF with Python3, but when you build the client it will rebuild TF and here you can't tell it
  to use Python3.
  If you do force Python3 then TF will compile correctly, but TFS doesn't support Python3 at all, it will try
  to use 2to3 automatic conversion that fails.
  There is no way, while building the client, to make so that TF is compiled for Python3 and TFS for Python2.
  If you modify BUILD file to build everything for Python3 the TF will still be build for Python2 and your client
  will fail at runtime like this:
  ```
  ImportError: /home/user/client/bazel-bin/my_api_gw.runfiles/org_tensorflow/tensorflow/python/_pywrap_tensorflow_internal.so: undefined symbol: PyClass_Type
  ```

- There is a package on Github that claims to be a simplified version of all this mess.
  It is not flexible enough, but would provide a good starting point for modification.
  Guess what, it doesn't work with recent versions of TF/TFS.
  https://github.com/sebastian-schlecht/tensorflow-serving-python

