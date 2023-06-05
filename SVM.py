
from sklearn import svm
# LabelEncoder, một tiện ích để chuyển nhãn được biểu thị dưới dạng chuỗi thành số nguyên
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocessing import SimplePreprocessor
from datasets import SimpleDatasetLoader
from imutils import paths
from nn import CNN_extract
import pickle

print("[INFO] Nạp ảnh...")
imagePaths = list(paths.list_images("datasets/face mask dataset new"))
image_size = (32, 32)
num_classes = 3
sp = SimplePreprocessor(32, 32) 
sdl = SimpleDatasetLoader(preprocessors=[sp]) 
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], image_size[0], image_size[1], 3))


print("[INFO] Dung lượng bộ nhớ chứa dữ liệu ảnh: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

# Chuyển nhãn từ chuỗi sang số
le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.15, random_state=2)
cnn_model = CNN_extract.build(width=32, height=32, depth=3, classes=3)

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("[INFO] Training the CNN model...")
cnn_model.fit(trainX, trainY)

print("[INFO] Extracting features using the CNN model...")
train_features = cnn_model.predict(trainX)
test_features = cnn_model.predict(testX)

print("[INFO] Đánh giá Bộ phân lớp SVM ...")

model = svm.SVC(kernel='rbf', gamma='scale' , max_iter=10000)

model.fit(train_features, trainY)

pickle.dump(model, open("SVM.model", 'wb'))

print(classification_report(testY, model.predict(test_features),target_names=["masked_weared_incorred", "with_mask", "without_mask"]))

