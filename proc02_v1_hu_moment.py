import cv2
import numpy as np
import urllib.request

def distance_to_similarity(distance):
    if distance <= 20:
        # 距離が20以下の場合は、類似度を80%以上とする
        return 80 + (20 - distance) * 1
    elif 20 < distance <= 40:
        # 距離が20を超え、40までの場合、類似度は徐々に80%から50%に低下
        return 80 - ((distance - 20) * (30 / 20))  # 20から40の範囲で80%から50%に減少
    elif 40 < distance <= 50:
        # 距離が40を超え、50までの場合、類似度は徐々に50%から0%に低下
        return 50 - ((distance - 40) * (50 / 10))  # 40から50の範囲で50%から0%に減少
    else:
        # 距離が40を超える場合は、類似度を0%とする
        return 0

# 画像のパス
path0 = 'question_01.png' 
path1 = 'Sample04.jpg' 

# OpenCVが理解できる形式に変換
image0 = cv2.imread(path0, cv2.IMREAD_COLOR)
image1 = cv2.imread(path1, cv2.IMREAD_COLOR)

# 画像のURL
#url0 = 'https://example.com/image1.jpg'
#url1 = 'https://example.com/image2.jpg'

# URLから画像データをメモリにダウンロード
# resp0 = urllib.request.urlopen(url0)
# resp1 = urllib.request.urlopen(url1)
# image0 = np.asarray(bytearray(resp0.read()), dtype="uint8")
# image1 = np.asarray(bytearray(resp1.read()), dtype="uint8")

# OpenCVが理解できる形式に変換
# image0 = cv2.imdecode(image0, cv2.IMREAD_COLOR)
# image1 = cv2.imdecode(image1, cv2.IMREAD_COLOR)

# グレースケール変換
gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# 二値化
_, binary0 = cv2.threshold(gray0, 127, 255, cv2.THRESH_BINARY_INV)
_, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY_INV)

# エッジ検出
edges0 = cv2.Canny(binary0, 100, 200)
edges1 = cv2.Canny(binary1, 100, 200)

# 形状記述子（Huの不変モーメント）を計算
moments0 = cv2.moments(edges0)
moments1 = cv2.moments(edges1)
huMoments0 = cv2.HuMoments(moments0).flatten()
huMoments1 = cv2.HuMoments(moments1).flatten()

# ログスケール変換
huMoments0 = -np.sign(huMoments0) * np.log10(np.abs(huMoments0))
huMoments1 = -np.sign(huMoments1) * np.log10(np.abs(huMoments1))

# 類似度の計算（ユークリッド距離を使用）
distance1 = distance_to_similarity(np.sqrt(np.sum((huMoments0 - huMoments1) ** 2)))

print(f'類似度スコア1（距離）: {distance1}')

