from jinja2 import Template, Environment, FileSystemLoader


class FileGenerator:
    def __init__(self, template_dir_path: str, template_styles: str):
        # 템플릿 파일이 있는 디렉토리 설정
        self.yaml_config = {}
        self.env = Environment(loader=FileSystemLoader(template_dir_path))

        # 템플릿 파일 로드
        self.template = self.env.get_template(template_styles)

    def render_template(self, data):
        return self.template.render(data)

    def save_to_file(self, rendered_text, file_name):
        with open(file_name, "w") as file:
            file.write(rendered_text)
        print(f"File saved as {file_name}")
