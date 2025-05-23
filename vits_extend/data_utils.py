import csv

def read_metadata_csv(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        for row in reader:
            if len(row) == 2:  # 假设文件格式为 文件名|wav音频对应文字|音频路径|期望生成音频的文字
                audio_path, text = row
                data.append({
                    'audio_path': audio_path,
                    'text': text
                })
    return data