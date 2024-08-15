import os
import argparse
import requests
import zipfile


def download_file(url, local_filename):
    """Скачивает файл из указанного URL и сохраняет его локально."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # фильтр пустых chunk
                    f.write(chunk)
    return local_filename


def unzip_file(zip_filepath, extract_to):
    """Распаковывает zip-файл в указанную директорию."""
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Скрипт для скачивания данных из Zenodo")
    parser.add_argument('--url', required=True, help='URL для скачивания данных')
    parser.add_argument('--output_dir', default='data', help='Директория для сохранения данных')
    parser.add_argument('--unzip', action='store_true', help='Распаковать архив после скачивания')
    args = parser.parse_args()

    # Создаём директорию для данных, если её нет
    os.makedirs(args.output_dir, exist_ok=True)

    # Определяем имя файла для сохранения
    local_filename = os.path.join(args.output_dir, os.path.basename(args.url))

    # Скачиваем файл
    print(f"Скачивание данных из {args.url}...")
    download_file(args.url, local_filename)
    print(f"Данные сохранены в {local_filename}")

    # Если необходимо, распаковываем файл
    if args.unzip and local_filename.endswith('.zip'):
        print(f"Распаковка архива {local_filename}...")
        unzip_file(local_filename, args.output_dir)
        print(f"Данные распакованы в {args.output_dir}")


if __name__ == '__main__':
    main()
