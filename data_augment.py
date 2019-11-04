import os
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array, array_to_img

# 画像を拡張する関数
def draw_images(generator, x, dir_name, index):
    save_name = 'extened-' + str(index)
    g = generator.flow(x, batch_size=1, save_to_dir=output_dir,
                       save_prefix=save_name, save_format='jpg')

    # 1つの入力画像から何枚拡張するかを指定（今回は50枚）
    for i in range(5000):
        bach = g.next()

# 出力先ディレクトリの設定
output_dir = "./gendata/"

if not(os.path.exists(output_dir)):
    os.mkdir(output_dir)

# 拡張する画像の読み込み
images = glob.glob(os.path.join("./data/", "*.jpg"))

# ImageDataGeneratorを定義
datagen = ImageDataGenerator(rotation_range=30,
                            width_shift_range=20,
							height_shift_range=0.,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            vertical_flip=True)
#:]
# 読み込んだ画像を順に拡張
for i in range(len(images)):
    img = load_img(images[i])
    img = img.resize((128, 128))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    draw_images(datagen, x, output_dir, i)
