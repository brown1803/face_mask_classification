from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import SimplePreprocessor
from datasets import SimpleDatasetLoader
from imutils import paths
import pickle  

# Lấy danh sách các ảnh trong folder
print("[INFO] Nạp ảnh...")
imagePaths = list(paths.list_images("datasets/face mask dataset new"))

sp = SimplePreprocessor(64, 64) 
sdl = SimpleDatasetLoader(preprocessors=[sp]) 
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 64 * 64 * 3)) 


print("[INFO] Dung lượng của bộ nhớ chứa dữ liệu ảnh: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

# Chuyển nhãn từ chuỗi sang số nguyễn
le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.15)

print("[INFO] Đánh giá Bộ phân lớp k-NN ...")

model = KNeighborsClassifier(n_neighbors=30,weights='distance')

model.fit(trainX, trainY)

pickle.dump(model, open("knn.model", 'wb'))

print(classification_report(testY, model.predict(testX),target_names=le.classes_))

