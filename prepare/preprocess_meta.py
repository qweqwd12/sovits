def read_meta_file(meta_file_path):
    data = []
    with open(meta_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 4:
                filename, audio_text, audio_path, target_text = parts
                data.append({
                    'filename': filename,
                    'audio_text': audio_text,
                    'audio_path': audio_path,
                    'target_text': target_text
                })
            if len(parts) == 4:
                filename, audio_text, audio_path, target_text, target_audio = parts
                data.append({
                    'filename': filename,
                    'audio_text': audio_text,
                    'audio_path': audio_path,
                    'target_text': target_text,
                    'target_audio': target_audio
                })
    return data