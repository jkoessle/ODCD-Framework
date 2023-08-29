import os
import argparse
import tensorflow as tf

import utils.prediction as pred


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parser = argparse.ArgumentParser("predict")
    parser.add_argument("--model-path", dest="model_path",
                        help="Specify which model to use.",
                        default="",
                        required=True,
                        type=str)
    parser.add_argument("--log-dir", dest="log_dir",
                        help="Specify directory of event logs to examine.",
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument("--encoding", dest="encoding",
                        help="Specify which encoding to use or was used.",
                        required=True,
                        type=str)
    parser.add_argument("--image-dir", dest="image_dir",
                        help="Specify where preprocessed images are stored and \
                            should be loaded from.",
                        default=None,
                        required=False,
                        type=str)
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Specify where to store results. \
                            If image-dir is not provided, logs are preprocessed and \
                                the resulting images are also stored here.",
                        required=True,
                        type=str)
    parser.add_argument("--cp-all", dest="cp_all",
                        help="Specify whether to identify all changepoints. \
                        Only needed for VDD encoding.",
                        default=None,
                        required=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--n-windows", dest="n_windows",
                        help="Number of windows. Only needed for WINSIM encoding.",
                        default=None,
                        required=False,
                        type=int)
    args = parser.parse_args()

    if args.image_dir is None:
        assert os.path.isdir(args.log_dir), "Please specify a valid event log \
            directory as an argument"
        image_dir = pred.prediction_pipeline(log_dir=args.log_dir,
                                             encoding_type=args.encoding,
                                             output_path=args.output_dir,
                                             cp_all=args.cp_all,
                                             n_windows=args.n_windows)
    else:
        image_dir = args.image_dir

    model = tf.saved_model.load(args.model_path)

    pred.predict(image_dir=image_dir,
                 output_path=args.output_dir,
                 model=model,
                 encoding_type=args.encoding,
                 n_windows=args.n_windows)
