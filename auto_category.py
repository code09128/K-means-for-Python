from sklearn.cluster import KMeans
import imutils
from imutils import paths
import numpy as np
#import argparse
import cv2

def describe(image):
    # convert the image to the L*a*b* color space, compute a histogram,
    # and normalize it
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [8,8,8], [0, 256, 0, 256, 0, 256])#獲取3D直方圖
    hist = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX).flatten()#將圖片大小標準化，忽略圖片大小對直方圖的影響
    # return the histogram
    return hist

#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True, help="path to the input dataset directory")
#ap.add_argument("-k", "--clusters", type=int, default=2, help="# of clusters to generate")
#args = vars(ap.parse_args())

# initialize the image descriptor along with the image matrix
data = []

# grab the image paths from the dataset directory
#imagePaths = list(paths.list_images(args["dataset"]))
path = r'D:\Dustin\AI\imageData\20200319_SortBoneScan\resizeMerge\20200504resize1000_1000\resizeMergeY' #路徑資料夾
imagePaths = list(paths.list_images(path))
imagePaths = np.array(sorted(imagePaths))

# loop over the input dataset of images
for imagePath in imagePaths:
        # load the image, describe the image, then update the list of data
        image = cv2.imread(imagePath)
        print(imagePath)
        image = imutils.resize(image, width = 350)
        hist = describe(image)
        data.append(hist)#描述符加入到資料集中

# cluster the color histograms
#clt = KMeans(n_clusters=args["clusters"], random_state=42)

#對描述符進行聚類
n = 1311
clt = KMeans(n_clusters=n, random_state=42)
labels = clt.fit_predict(data)

# loop over the unique labels
for label in np.unique(labels):
        # grab all image paths that are assigned to the current label
        # 獲取每個叢集的唯一ID,進行分類
        labelPaths = imagePaths[np.where(labels == label)]

        # loop over the image paths that belong to the current label
        # 將同一叢集的圖片輸出顯示
        for (i, path) in enumerate(labelPaths):
                # load the image and display it
                image = cv2.imread(path)
                image = imutils.resize(image, width = 250)
                # cv2.imshow("Cluster {}, Image #{}".format(label + 1, i + 1), image)

                print("Cluster",(label + 1),"Image",path)
                print("Cluster",(label + 1),"Image",path,file=open(r"D:\Dustin\AI\KNN\kmeans-office-master\result.csv", "a"), flush=True)

        # wait for a keypress and then close all open windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

