from PIL import Image
import os

# 元の画像が保存されているディレクトリパス
input_directory = r'C:\Users\hawke\OneDrive\デスクトップ\python\cerema\change_clothes\data\person'

# リサイズ後の画像を保存するディレクトリパス
output_directory = r'C:\Users\hawke\OneDrive\デスクトップ\python\cerema\change_clothes\data\512\person'

# 512x512のサイズ
target_size = (512, 512)

# 入力ディレクトリ内の画像ファイルを処理
for filename in os.listdir(input_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # 画像のフルパス
        image_path = os.path.join(input_directory, filename)

        # 画像を開く
        with Image.open(image_path) as img:
            # 画像をリサイズ
            resized_img = img.resize(target_size, Image.ANTIALIAS)

            # 出力ディレクトリに保存
            output_path = os.path.join(output_directory, filename)
            resized_img.save(output_path)

print("リサイズが完了しました。")
