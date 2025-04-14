import cv2
import numpy as np
from PIL import Image

# 画像のタイルから指定されたサイズ分を追加でトリミングする関数
def additional_trim(tile, trim_size=1):
    # トリミング後の新しい境界を計算する
    left, top = trim_size, trim_size
    right = tile.size[0] - trim_size
    bottom = tile.size[1] - trim_size

    # 負の寸法を持たないようにする
    left, top = max(0, left), max(0, top)
    right, bottom = max(left, right), max(top, bottom)

    # 新しい寸法で画像をトリミングする
    return tile.crop((left, top, right, bottom))

# 黒い枠線を持つタイルを見つけ、1ピクセル追加でトリミングする関数
def trim_to_black_corners_with_additional_trim(image_path, squares):
    trimmed_image_paths = []

    # 全体のシート画像を読み込む
    sheet_image = Image.open(image_path)

    for i, (left, top, right, bottom) in enumerate(squares):
        # シートからタイルを切り出す
        tile = sheet_image.crop((left, top, right, bottom))
        tile_np = np.array(tile)

        # グレースケールに変換し、黒が前景の二値画像に変換する
        gray = cv2.cvtColor(tile_np, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # 二値画像で輪郭を見つける
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 最大の輪郭が枠線であると仮定し、そのバウンディングボックスを見つける
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # 輪郭の矩形を調整して角が黒になるようにする
            while x < w - 1 and y < h - 1 and np.mean(tile_np[y, x]) > 200:
                x += 1
                y += 1
            while x < w - 1 and y < h - 1 and np.mean(tile_np[y + h - 1, x + w - 1]) > 200:
                w -= 1
                h -= 1

            # 調整されたバウンディングボックスに画像をトリミングする
            trimmed_tile = tile.crop((x, y, x + w, y + h))
            
            # 1ピクセル追加でトリミングする
            trimmed_tile = additional_trim(trimmed_tile)

            # トリミングされた画像を保存する
            row_number = i // 4 + 1
            column_number = i % 4 + 1
            trimmed_path = f'C:\\Users\\kazuk\\Desktop\\20240331_JapanClubUnity\\sample_20240704_01_{row_number}_{column_number}.png'
            trimmed_tile.save(trimmed_path)
            trimmed_image_paths.append(trimmed_path)

    return trimmed_image_paths

# 追加のトリミング処理を定義
def trim_all_edges_to_white(tile):
    tile_np = np.array(tile)
    left, top, right, bottom = 0, 0, tile.size[0], tile.size[1]

    while np.mean(tile_np[top, left]) <= 200:
        top += 1
        left += 1
    while np.mean(tile_np[bottom - 1, right - 1]) <= 200:
        bottom -= 1
        right -= 1
    while np.mean(tile_np[top, right - 1]) <= 200:
        top += 1
        right -= 1
    while np.mean(tile_np[bottom - 1, left]) <= 200:
        bottom -= 1
        left += 1

    return tile.crop((left, top, right, bottom))


# 画像を読み込む
image_path = 'C:\\Users\\kazuk\\Desktop\\20240331_JapanClubUnity\\sample_20240704_01.png'

# 与えられた枠の頂点情報（左上のx座標、左上のy座標、右下のx座標、右下のy座標）
provided_coordinates = [
    (60, 303, 392, 635),
    (390, 303, 722, 635),
    (722, 303, 1054, 635),
    (1054, 303, 1386, 635),
    (60, 740, 392, 1072),
    (390, 740, 722, 1072),
    (722, 740, 1054, 1072),
    (1054, 740, 1386, 1072),
    (60, 1167, 392, 1499),
    (390, 1167, 722, 1499),
    (722, 1167, 1054, 1499),
    (1054, 1167, 1386, 1499),
    (60, 1606, 392, 1938),
    (390, 1606, 722, 1938),
    (722, 1606, 1054, 1938),
    (1054, 1606, 1386, 1938)
]

# 正確なタイルをトリミングして黒い枠線までトリミングする処理を実行
trimmed_image_paths = trim_to_black_corners_with_additional_trim(image_path, provided_coordinates)

# 元のトリミングされた画像に対して追加のトリミング処理を適用する
original_trimmed_paths = trimmed_image_paths
final_trimmed_paths = []

for path in original_trimmed_paths:
    tile = Image.open(path)
    final_trimmed_tile = trim_all_edges_to_white(tile)
    final_trimmed_path = path.replace("one_px_trimmed", "final_trimmed")
    final_trimmed_tile.save(final_trimmed_path)
    final_trimmed_paths.append(final_trimmed_path)

# トリミングされた画像のパスを出力
print(final_trimmed_paths)

