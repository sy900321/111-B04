import cv2
import numpy as np
import json

# 讀取COCO JSON標註文件
with open('odm_polygon_anntotaion.json', 'r') as f:
    annotations = json.load(f)

# 遍歷每張圖片
for image_info in annotations['images']:
    # 讀取圖像路徑
    image_path = image_info['file_name']

    # 讀取圖像
    image = cv2.imread(image_path)

    # 建立空的遮罩
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # 遍歷每個標註
    for annotation in annotations['annotations']:
        # 檢查標註是否屬於當前圖像
        if annotation['image_id'] == image_info['id']:
            # 取得多邊形頂點座標
            polygon = annotation['segmentation'][0]

            # 將多邊形頂點座標轉換為NumPy陣列
            points = np.array(polygon, np.int32).reshape((-1, 2))

            # 在遮罩上繪製多邊形
            cv2.fillPoly(mask, [points], color=(255))

    # 將遮罩應用到原始圖像上
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    # 顯示結果

    '''
    img = cv2.resize(image, None, fx=0.1, fy=0.1)
    cv2.imshow('Original Image', img)
    
    sim = cv2.resize(segmented_image, None, fx=0.1, fy=0.1)
    cv2.imshow('Segmented Image', sim)
    
    ma = cv2.resize(mask, None, fx=0.1, fy=0.1)
    cv2.imshow(image_path, ma)
    '''
    
    cv2.imwrite("segmented/" + image_path, mask)
    #cv2.waitKey(0)

cv2.destroyAllWindows()
