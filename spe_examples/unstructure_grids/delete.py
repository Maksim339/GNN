import os


def delete_files_with_prefix(prefix, directory="."):
    """
    Удаляет все файлы в указанной директории, начинающиеся с заданного префикса.
    """
    deleted_files = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
                deleted_files.append(filename)

    if deleted_files:
        print(f"Удалены файлы: {', '.join(deleted_files)}")
    else:
        print(f"Файлы с префиксом '{prefix}' не найдены в директории '{directory}'.")


if __name__ == "__main__":
    delete_files_with_prefix("two_phase_step_")
