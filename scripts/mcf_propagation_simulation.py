import os
import subprocess


def download_laser_data():
    # Определяем путь к корневой директории проекта
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Определяем путь к скрипту download_data.py
    download_script = os.path.join(project_root, 'data', 'download_data.py')

    url = 'https://zenodo.org/record/1234567/files/mcf_propagation_data.zip?download=1'
    output_dir = os.path.join(project_root, 'data', 'mcf_propagation')

    # Формируем команду для скачивания данных
    cmd = [
        'python', download_script,
        '--url', url,
        '--output_dir', output_dir,
        '--unzip'
    ]

    # Запускаем процесс скачивания
    subprocess.run(cmd, check=True)


def main():
    download_laser_data()
    # Добавьте код для запуска симуляции после скачивания данных


if __name__ == '__main__':
    main()
