from typing import List, Tuple
import onnx


class DrpaiYamlGeneratorError(Exception):
    def __init__(self, message="Error in DrpaiYamlGenerator"):
        self.message = message
        super().__init__(self.message)


class DrpaiYamlGenerator:
    def __init__(self, model_path: str):
        self.model = onnx.load(model_path)
        self.yaml_config = {}

    def generate_input_config(self):
        input_info = self.model.graph.input

        input_body = []
        input_preprocess = []

        for idx, input in enumerate(input_info):
            input_name, hwc_input_shape = self.__convert_nchw_to_hwc(input)

            # 형식에 맞는 문자열 생성
            input_body_str = f"  -\n    name: \"{input_name}\"\n    format: \"RGB\"\n    order: \"HWC\"\n    shape: [{', '.join(hwc_input_shape)}]\n    type: \"fp16\""
            input_preprocess_str = f'  -\n    src      : ["camera_data"]\n    dest     : ["{input_name}"]\n\n    operations:\n    -\n      op: memcopy\n      param:\n        WORD_SIZE : 2 # FP16'

            input_body.append(input_body_str)
            input_preprocess.append(input_preprocess_str)

        all_input_body = "\n".join(input_body)
        all_input_preprocess = "\n".join(input_preprocess)

        # 요 부분은 리사이즈를 넣는게 낳을수도 있음
        # 모든 input을 받으려면...
        input_shape = [int(dim_value) for dim_value in hwc_input_shape]

        self.yaml_config["input_body"] = all_input_body
        self.yaml_config["input_shape"] = input_shape
        self.yaml_config["input_preprocess"] = all_input_preprocess

    def generate_output_config(self):
        output_info = self.model.graph.output

        output_body = []
        output_post = []
        output_postprocess = []

        for idx, input in enumerate(output_info):
            output_name, hwc_output_shape = self.__convert_nchw_to_hwc(input)

            # 형식에 맞는 문자열 생성
            output_body_str = f"  -\n    name: \"{output_name}\"\n    shape: [{', '.join(hwc_output_shape)}]\n    order: \"HWC\"\n    type: \"fp16\""
            output_post_str = f"  -\n    name: \"post_{output_name}\"\n    shape: [{', '.join(hwc_output_shape)}]\n    order: \"HWC\"\n    type: \"fp16\""
            output_postprocess_str = f'  -\n    src: ["{output_name}"]\n    dest: ["post_{output_name}"]\n\n    operations:\n    -\n      op: memcopy\n      param:\n        WORD_SIZE : 2 # FP16\n'

            output_body.append(output_body_str)
            output_post.append(output_post_str)
            output_postprocess.append(output_postprocess_str)

        all_output_body = "\n".join(output_body)
        all_output_post = "\n".join(output_post)
        all_output_postprocess = "\n".join(output_postprocess)

        self.yaml_config["output_body"] = all_output_body
        self.yaml_config["output_post"] = all_output_post
        self.yaml_config["output_postprocess"] = all_output_postprocess

    def __convert_nchw_to_hwc(
        self, input_obj: onnx.ValueInfoProto
    ) -> Tuple[str, List[str]]:
        """
        Converts an input shape from NCHW to HWC format.

        Parameters:
        input_obj (object): An object with 'name' attribute and a 'type.tensor_type.shape.dim' attribute.

        Returns:
        tuple: A tuple containing the input name and the converted HWC shape.
        """
        input_name = input_obj.name
        nchw_input_shape = [
            str(dim.dim_value) for dim in input_obj.type.tensor_type.shape.dim
        ]

        if len(nchw_input_shape) == 4:
            # Convert NCHW to HWC
            hwc_input_shape = [
                nchw_input_shape[2],
                nchw_input_shape[3],
                nchw_input_shape[1],
            ]
        elif len(nchw_input_shape) == 3:
            # Convert CHW to HWC (assuming the first dimension is channel)
            hwc_input_shape = [
                nchw_input_shape[1],
                nchw_input_shape[2],
                nchw_input_shape[0],
            ]
        else:
            raise DrpaiYamlGeneratorError(
                "The model's input is not in NCHW or CHW format."
            )

        return input_name, hwc_input_shape
