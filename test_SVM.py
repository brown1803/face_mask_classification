from preprocessing import SimplePreprocessor 
from datasets import simpledatasetloader 
from nn import CNN_extract
import cv2
import pickle 

classLabels = ['mask_weared_incorrect','with_mask','without_mask']

print("[INFO] Đang nạp ảnh để cho bộ phân lớp dự đoán...")

image = cv2.imread("Image_test/mc.jpeg")

sp = SimplePreprocessor(32, 32)
image = image.reshape((32, 32, 3))
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp])

data, _ = sdl.load(['Image_test/incorrect_mask.jpeg'])

data = data.reshape((data.shape[0], 3072))
# sp = SimplePreprocessor(32, 32)
# image = sp.preprocess(image)
# image = image.reshape((1, 32, 32,3))

# sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp])

# cnn_model = CNN_extract.build(width=32, height=32, depth=3, classes=3)
# features = cnn_model.predict(image)
# features = features.reshape((features.shape[0], -1))

# data, _ = sdl.load(["Image_test/incorrect_mask.jpeg"])

# data = data.reshape((data.shape[0], 3072))

print("[INFO] Nạp model SVM ...")
model = pickle.load(open('SVM.model', 'rb'))

# Dự đoán
print("[INFO] Thực hiện dự đoán ảnh để phân lớp...")
preds = model.predict(image) 

print(preds)

# Đọc file ảnh
image = cv2.imread("Image_test/mc.jpeg")

# Viết label lên ảnh
cv2.putText(image, "label: {}".format(classLabels[preds[0]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Hiển thị ảnh
cv2.imshow("Image", image)
cv2.waitKey(0)
