from business_logic.template_file_generator import FileGenerator
from business_logic.drpai_yaml_generator import DrpaiYamlGenerator
from typing import List, Tuple
from pathlib import Path
import os
import subprocess


def get_file_paths_and_names_in_dir(
    directory_path: str,
) -> Tuple[List[Path], List[str]]:
    """디렉토리의 파일 정보 가져오기"""

    if not Path(directory_path).is_dir():
        raise ValueError(f"Provided path '{directory_path}' is not a valid directory.")
    file_names = []
    file_paths = []
    try:
        for root, dirs, files in os.walk(directory_path):
            for name in files:
                full_path = Path(root) / name
                file_names.append(full_path.name)
                file_paths.append(full_path)
    except Exception as e:
        raise RuntimeError(f"Error occurred while accessing files in directory: {e}")
    return file_names, file_paths


def generate_yaml_for_performance_evaluation(
    model_path: str, save_file_name="output_file.yaml"
):
    """drp-ai 성능 측정용 drp-ai 템플릿 생성"""

    # pre/post를 최소화한 성능 측정용 drp-ai 템플릿 선택
    template_dir_path = "./template"
    template_styles = "performance_evaluation_template.yaml"

    try:
        file_generator = FileGenerator(template_dir_path, template_styles)
    except Exception as e:
        raise RuntimeError(f"FileGenerator 초기화 중 에러 발생: {e}")

    try:
        # 템플릿에 모델의 정보(input/output) 입력
        drp_generator = DrpaiYamlGenerator(model_path)
        drp_generator.generate_input_config()
        drp_generator.generate_output_config()
        # 템플릿 렌더링
        rendered = file_generator.render_template(drp_generator.yaml_config)
        # 파일로 저장
        file_generator.save_to_file(rendered, save_file_name)
    except Exception as e:
        raise RuntimeError(f"DRP-AI YAML 생성 중 에러 발생: {e}")


def validate_model_conversion(drpai_tool_path, drpai_model_name) -> bool:
    """onnx -> drpai 변환 결과 확인"""

    # 변환 결과가 저장되는 폴더
    translator_output_path = os.path.join(drpai_tool_path, "output", drpai_model_name)

    if not Path(translator_output_path).is_dir():
        raise ValueError(
            f"Provided path '{translator_output_path}' is not a valid directory."
        )

    file_list = os.listdir(translator_output_path)
    translator_result_file_count = len(file_list)

    # Model Conversion Successful
    if translator_result_file_count == 18:
        return True
    # Model Conversion Failure
    else:
        return False


def quantize_with_renesas(
    drpai_quantize_path: str,
    input_model_path: str,
    output_model_path: str,
    calibrate_dataset: str,
    norm_mean: str,
    norm_std: str,
    datareader_path: str,
):
    """onnx(fp32) -> onnx(int8) 변환"""

    cmd = f"""
    python3 -m drpai_quantizer.cli_interface \
    --input_model_path {input_model_path} \
    --output_model_path {output_model_path} \
    --norm_mean {norm_mean} \
    --norm_std {norm_std} \
    --calibrate_method MinMax \
    --calibrate_dataset {calibrate_dataset} \
    --datareader_path {datareader_path}
    """

    print("Executing quantize command:", cmd)
    try:
        subprocess.run(cmd, cwd=drpai_quantize_path, shell=True, check=True, text=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


def convert_onnx_to_drpai(
    drpai_tool_path, drpai_model_name, onnx_path, yaml_path, s_addr="0x80000000"
) -> bool:
    """onnx -> drpai 변환"""

    cmd = f"cd {drpai_tool_path} && ./run_Translator_v2h.sh {drpai_model_name} --onnx {onnx_path} --prepost {yaml_path} --s_addr {s_addr}"
    print("Executing tranlator command:", cmd)

    try:
        subprocess.run(cmd, shell=True, check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

    return validate_model_conversion(drpai_tool_path, drpai_model_name)


def print_conversion_results(convert_result_list):
    """onnx->drpai 모델 변환 결과를 출력"""
    print()
    RED = "\033[91m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    print("-------- Convert Result --------")
    for model_name, result in convert_result_list:
        print(f"{model_name}, ", end="")
        if result:
            print(GREEN + "Succeeded" + RESET)
        else:
            print(RED + "Failed" + RESET)
    print("--------------------------------")
    print()


if __name__ == "__main__":
    # TOOL 경로 셋팅
    DRPAI_QUANTIZE_PATH = os.getenv("DRPAI_QUANTIZE_PATH")
    DRPAI_TRANSLATOR_PATH = os.getenv("DRPAI_TRANSLATOR_PATH")

    # translator option
    yaml_file_path = "/home/PyGeniusTemplates/"
    yaml_file_name = "output_file.yaml"
    yaml_path = yaml_file_path + yaml_file_name
    s_addr = "0x80000000"

    # quantize option
    datareader_path = "/home/drp-ai_work/tutorials/Translate/nchw_datareader_yolox_s.py"
    calibrate_dataset = "/home/drp-ai_work/tutorials/Translate/calibrate_sample"
    calibrate_method = "MinMax"
    norm_mean = "[0.0,0.0,0.0]"
    norm_std = "[1.0,1.0,1.0]"

    # yaml_path = "/home/PyGeniusTemplates/renesas_prepost_renesas_YoloX_S.yaml"
    # yaml_path = "/home/PyGeniusTemplates/renesas_prepost_nota_YoloX_S.yaml"
    # yaml_path = "/home/PyGeniusTemplates/renesas_prepost_nota_original_YoloX.yaml"
    # yaml_path = "/home/PyGeniusTemplates/renesas_prepost_nota_ModelZOO_YoloX_S.yaml"
    # yaml_path = "/home/PyGeniusTemplates/renesas_prepost_nota_JH_YoloX_S.yaml"
    # yaml_path = "/home/PyGeniusTemplates/renesas_prepost_nota_Core_YoloX_S.yaml"
    yaml_path = (
        "/home/PyGeniusTemplates/renesas_prepost_wonjin_np_trainer_np_Core_YoloX_S.yaml"
    )
    print(
        f"################ 현재 자동생성한 yaml이 아닌 임시로 Rensas와 똑같이 성능 측정 하기 위해 {yaml_path} 파일 쓰고 있음######### "
    )

    convert_result_list = []

    dir_path = "/home/PyGeniusTemplates/models"
    quantize_output_dir_path = "/home/PyGeniusTemplates/quantize_models/"
    file_names, file_paths = get_file_paths_and_names_in_dir(dir_path)

    for model_name, model_path in zip(file_names, file_paths):
        generate_yaml_for_performance_evaluation(
            model_path, save_file_name=yaml_file_name
        )

        model_name = model_name.replace(".onnx", "_INT8.onnx")
        q_output_path = quantize_output_dir_path + model_name

        # 르네사스 Quantize
        quantize_with_renesas(
            drpai_quantize_path=DRPAI_QUANTIZE_PATH,
            input_model_path=str(model_path),
            output_model_path=str(q_output_path),
            calibrate_dataset=calibrate_dataset,
            norm_mean=norm_mean,
            norm_std=norm_std,
            datareader_path=datareader_path,
        )

        # onnx(int8) -> drp-ai 변환
        model_name = model_name.replace(".onnx", "")
        convert_result = convert_onnx_to_drpai(
            drpai_tool_path=DRPAI_TRANSLATOR_PATH,
            drpai_model_name=model_name,
            onnx_path=q_output_path,
            yaml_path=yaml_path,
            s_addr=s_addr,
        )
        convert_result_list.append((model_name, convert_result))

    # 결과 출력
    print_conversion_results(convert_result_list)
