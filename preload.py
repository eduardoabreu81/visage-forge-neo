def preload(parser):
    parser.add_argument(
        "--visage-models-dir",
        type=str,
        default=None,
        help="Folder for Visage ONNX models and face models (default: models/visage)",
    )
