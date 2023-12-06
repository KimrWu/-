import cv2  #導入 OpenCV（Open Source Computer Vision Library）庫
import numpy as np  #導入 NumPy 庫 NumPy 縮寫為 np
import mediapipe as mp  #導入 MediaPipe 庫 MediaPipe 縮寫為 mp
import random   #導入 Random 庫
import pygame   #導入 PyGame 庫
import time #導入 Time 庫
import os

# 初始化Pygame
pygame.init()
# 設置窗口尺寸
win_width, win_height = 1700, 1000  #定義視窗的寬度和高度
win = pygame.display.set_mode((win_width, win_height))  #創建了一個名為 "win" 的 Pygame 窗口，並使用 pygame.display.set_mode() 函數來設定視窗的大小。視窗的大小由前面定義的 win_width 和 win_height 決定
pygame.display.set_caption("Rock-Paper-Scissors Game")  #設定了視窗的標題，用於讓使用者知道視窗顯示的是什麼內容
font = pygame.font.SysFont('arialblack', 60)    #創建一個指定字體（例如 "Arial Black"）和字號（例如 60）的字體對象


#更方便地使用 MediaPipe 提供的功能來進行手部姿勢檢测
mp_drawing = mp.solutions.drawing_utils #將 mp.solutions.drawing_utils 賦值給變量 mp_drawing
mp_hands = mp.solutions.hands   #將 mp.solutions.hands 賦值給變量 mp_hands

#加載八個不同的圖像文件到八個變量中
paper_image = cv2.imread("paper.png")
scissor_image = cv2.imread("scissor.png")
stone_image = cv2.imread("stone.png")

win_image = cv2.imread("win_Flip Image.png")
win_image1 = cv2.imread("win.png")
lose_image = cv2.imread("YOU_LOSE.jpg")
lose_image1 = cv2.imread("YOU_LOSE_Flip Image.jpg")
tie_image = cv2.imread("TIE.jpg")

# 對於攝影機的輸入:
cap = cv2.VideoCapture(0)   #創建了一個視訊捕獲對象，它用於訪問視訊攝像頭
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  #使用 cap 對象的 .get() 方法來獲得當前視頻幀的寬度，並將其存儲在變數 frameWidth 中
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)    #使用 cap 對象的 .get() 方法來獲得當前視頻幀的高度，並將其存儲在變數 frameHeight 中

computer_mode = "random"    #模式模認為"隨機模式"

glegr = 'nothing'   #手勢標籤默認為'無'
#以上為設定導入部分
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#使用 mp_hands.Hands 函數創建一個手部檢測器的實例，並將其命名為 hands
#這個函數初始化了一個可以用於檢測手部姿勢的物件，並設定了兩個參數，min_detection_confidence 和 min_tracking_confidence，用於設定檢測手部的信心閾值(有效的最低可接受置信度（confidence）閥值)
with mp_hands.Hands(
        min_detection_confidence=0.7,   #當偵測結果的置信度得分高於或等於0.7時，結果才會被視為有效。detection:確定是否檢測到手部;tracking:確定手部在幀之間的穩定性
        min_tracking_confidence=0.7) as hands:  #會在離開with(資源管理器)後正確釋放資源
    # 創建一個空白圖像"computer_result"，用於存儲計算機的結果，尺寸與攝像頭視頻幀的尺寸相同
    computer_result = np.zeros((int(frameHeight), int(frameWidth), 3), dtype=np.uint8)  #np.zeros 是 NumPy 函數，用於創建一個全為零的數組，這個數組具有高度 frameHeight、寬度 frameWidth 並且具有 3 個通道（對應到 RGB 顏色通道），dtype=np.uint8 指定了數組中元素的數據類型為 8 位無符號整數，這是通常用於表示圖像像素的數據類型
    # 創建一個空白圖像"computer_result1"，用於存儲計算機的結果，尺寸與攝像頭視頻幀的尺寸相同
    computer_result1 = np.zeros((int(frameHeight), int(frameWidth), 3), dtype=np.uint8)
    # 創建一個空白圖像"computer_result2"，用於存儲計算機的結果，尺寸與攝像頭視頻幀的尺寸相同
    computer_result2 = np.zeros((int(frameHeight), int(frameWidth), 3), dtype=np.uint8)

    prev_linearity = None  # 初始化前一幀的 linearity
    #text1 = 'nothing'
    #確保攝像頭處於打開狀態時，持續從攝像頭中讀取影像幀，如果無法讀取有效的幀，則忽略該幀並繼續等待下一個有效的幀
    while cap.isOpened():   #它將一直運行，如果攝像頭無法打開或出現錯誤，返回 False，這樣就可以退出循環
        success, image = cap.read() #從攝像頭捕獲的一幀影像，cap.read() 函數返回兩個值，第一個值是一個布林值（success），用於指示讀取是否成功，第二個值是讀取的影像幀（image）。
        if not success: #檢查讀取是否成功，如果 success 是 False，這意味著無法讀取有效的影像幀
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue    #continue 是一個控制流程語句，如果無法讀取有效的影像幀，continue 將使程式跳過當前的循環迭代，並繼續下一次循環，而不執行後續的程式碼。

        #將攝像頭捕獲的影像進行處理，以便它可以被MediaPipe庫使用，並且後續的操作不會修改原始影像。這樣就可以進行手部姿勢檢測
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) #cv2.flip(image, 1)：用於對影像進行水平翻轉，數字1表示水平翻轉，0表示垂直翻轉，-1 表示同時水平和垂直翻轉
        #cv2.cvtColor(...)：用於將影像的顏色空間轉換，在這裡，它將影像從BGR（藍綠紅）色彩空間轉換為RGB（紅綠藍）色彩空間。MediaPipe庫預期的輸入影像是RGB格式。

        image.flags.writeable = False   #將影像的"writeable"屬性設置為False，為了確保後續的操作不會意外地修改影像，以防止出現錯誤

        # 使用MediaPipe庫的手部姿勢模型（hands）對處理後的影像進行處理，從而檢測出影像中的手部姿勢。results 變數將包含檢測到的結果，包括手部的關鍵點位置和連接信息。
        results = hands.process(image)

        #將處理後的影像轉換為BGR色彩空間，以便它可以正確地在顯示窗口中顯示出來，可以確保影像以正確的色彩顯示給用戶觀看
        image.flags.writeable = True    #需要將其設置回True，以便對影像進行後續的修改
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  #將其轉回BGR，因為OpenCV通常使用BGR顏色順序來顯示影像
        #以上為讀取攝像頭並處理的部分
        #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        #處理檢測到的手部姿勢，並在影像上繪製手部關鍵點和相應的文字標籤，這對於手部姿勢分析和視覺化非常有用。
        if results.multi_hand_landmarks:    #檢查是否檢測到了多個手部關鍵點，如果有，則執行下面的操作，這個檢查是為了確保在繼續處理手部關鍵點之前，至少要檢測到一隻手。
            for hand_landmarks in results.multi_hand_landmarks: #一個迴圈，用於遍歷每一個檢測到的手部關鍵點，使你能夠獲得每隻手部的關鍵點信息以進行後續處理

                # draw_landmarks函數將手部姿勢的關鍵點和連接線繪製在影像上，mp_drawing 是 MediaPipe 提供的一個工具，用於繪製不同姿勢的關鍵點和連接線
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                finger_points = [[], [], [], [], []]    #創建了一個名為 finger_points 的空列表，用於存儲手指的位置信息，包含五個子列表，各用於存儲五個手指的位置
                for id, lm in enumerate(hand_landmarks.landmark):   #一個迴圈，它遍歷 hand_landmarks.landmark 這個列表，其中每個元素代表手部關鍵點的信息。在這個迴圈中，每次迭代都會獲得兩個值：id 和 lm。
                # id 是關鍵點的索引或標識符，通常是一個整數值，用於區分不同的關鍵點。例如，0 可能代表手部的基本位置，1 可能代表食指的位置，2 可能代表中指的位置，以此類推
                # lm 是代表關鍵點的具體信息的變數，可能包含有關該關鍵點的坐標、深度、置信度等信息，具體取決於代表手部關鍵點的數據結構
                # 在迴圈內，你可以使用這兩個值來訪問和處理每個手部關鍵點的信息。可以對每個關鍵點執行相應的操作，例如獲取其坐標，計算特定特徵，或者繪製圖形等

                    h, w, c = image.shape   #image.shape 返回一個包含圖像尺寸和通道數信息的元組，h: 圖像的高度;w: 圖像的寬度;c: 圖像的通道數，通常是 1（對於灰度圖像）或 3（對於彩色圖像）。這個程式碼將圖像的屬性解包到這三個變數中，以便後續的處理。
                    cx, cy = int(lm.x * w), int(lm.y * h)   #計算出關鍵點在影像中的座標，其中 lm.x 和 lm.y 是關鍵點的相對座標（介於0和1之間），表示關鍵點在圖像中的相對位置，而 w 和 h 是影像的寬度和高度，將相對位置轉換為實際的像素位置。這樣做之後，cx 和 cy 將分別是手部關鍵點的實際像素坐標，這樣就可以在圖像上進行繪製或其他相關處理。
                    cv2.circle(image, (cx, cy), 10, (255, 0, 0), 2) #繪製一個藍色的圓圈，表示手部姿勢中的關鍵點位置。使用 cv2.circle 函數，其中 (cx, cy) 是圓圈的中心座標，10 是圓的半徑，(255, 0, 0) 是藍色的BGR顏色碼，2 是圓圈的線寬
                    if id != 0: #檢查是否處理的是非零號（非手掌部位）的關鍵點，通常，手部關键點中的第一個點（0號點）代表手部的底部或手腕，而其他點代表手部的其他部位，如手指的末端或關節。只對其他手部關键點執行特定操作，因為手掌部位不是手指，所以只需要關心手指的位置
                        finger_points[(id - 1) // 4].append([cx, cy])   #根據關鍵點的ID（0到20之間的整數）將座標 [cx, cy] 添加到相應手指的子列表中。id - 1 是因為ID的範圍是1到20，而我們想將它們映射到五個手指中之一。// 4 是整數除法，用於將ID映射到手指的索引
                    # 將關鍵點的ID以文本的形式顯示在影像上，使用 cv2.putText 函數，其中 str(id) 是要顯示的文本，(cx, cy) 是文本的位置，cv2.FONT_HERSHEY_SIMPLEX 是字體，1 是文字的大小縮放因子，(0, 255, 255) 是黃色的BGR顏色碼，1 是文字的線寬，cv2.LINE_AA 是用於抗鋸齒的樣式
                    cv2.putText(image, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)


                #一個名為check_colinearity的函數，用於檢查一組點是否共線，通常用於分析手部姿勢中手指的排列是否共線
                def check_colinearity(points, tolerance=0.3):   #points：一個包含多個點坐標的二維數組，tolerance（可選）：用於確定在多大程度上點被視為共線。預設值為0.3。
                    vectors = np.diff(points, axis=0)   #np.diff() 函數用於計算陣列中相鄰元素的差值，axis=0 表示沿著陣列的第一個軸（通常是垂直方向）計算差值，將這些向量存儲在vectors中

                    #計算相鄰向量之間的夾角的正弦（sin），這是通過計算兩個向量的點積（dot product），並根據向量的模長計算的，並將計算的正弦值存儲在sins列表中
                    sins = []   #創建一個空列表 sins，它將用於存儲計算出的正弦值
                    for i in range(2, vectors.shape[0] + 1):    #使用一個 for 迴圈遍歷 vectors 陣列的元素，從 2 開始直到 vectors 陣列的最後一個元素
                        dot = np.dot(vectors[i - 2], vectors[i - 1])    #計算 dot，這是兩個向量的點積。vectors[i - 2] 和 vectors[i - 1] 分別是 vectors 陣列中的前一個和當前向量
                        norm_i_2 = np.linalg.norm(vectors[i - 2])   #計算一個指定向量的歐幾里德範數，也就是該向量的模長
                        norm_i_1 = np.linalg.norm(vectors[i - 1])   #計算一個指定向量的歐幾里德範數，也就是該向量的模長
                        cos = dot / (norm_i_2 * norm_i_1)   #兩個向量之間的夾角的餘弦值，通過將 dot 除以 norm_i_2 和 norm_i_1 之積來計算
                        sin = np.sqrt(1 - cos ** 2) #兩個向量之間的夾角的正弦值，通過從 1 減去 cos 的平方根來計算
                        sins.append(sin)    #將 sin 添加到 sins 列表中，如果 sin 值接近零，這意味著這兩個向量接近共線

                    return np.all(np.abs(sins) < tolerance) #使用np.all函數來檢查sins列表中的所有正弦值是否都小於給定的容忍值tolerance。如果是，則返回True，表示所有點共線；否則返回False，表示它們不共線。

                #根據手指的排列情況，選擇相應的手勢模式，然後將相應的圖像覆蓋到畫面上，以實現互動效果
                finger_points = np.array(finger_points)#將finger_points列表轉換為NumPy數組，可以更方便地進行數學和數组操作，finger_points包含了每個手指的坐標點
                # 創建了一個名為linearity的列表，其中包含五個元素，表示每個手指的排列情況。check_colinearity 函數用於檢查一組座標點是否共線，如果某個手指的點共線，對應的元素為1，否則為0。
                linearity = [1 if check_colinearity(finger_points[i]) else 0 for i in range(5)]

                paper_pattern = [1, 1, 1, 1, 1] #紙（paper）的手勢中，所有手指都應該共線，所以對應模式是全1
                scissor_pattern = [0, 1, 1, 0, 0]   #剪刀（scissor）的手勢中，第二和第三個手指應該共線，所以模式是[0, 1, 1, 0, 0]
                stone_pattrn = [0, 0, 0, 0, 0]  #石頭（stone）的手勢中，所有手指都不應共線，所以模式是全0
                patterns = [paper_pattern, scissor_pattern, stone_pattrn]   #這個列表包含了所有可能的手勢模式
                images = [scissor_image, stone_image, paper_image]

                distance = [np.linalg.norm(np.array(linearity) - np.array(pattern)) for pattern in patterns]    #計算了每種手勢（patterns 中的每個元素）與目前偵測到的手勢之間的距離。具體來說，它計算了兩個向量之間的歐氏距離，並將這些差異存儲在distance列表中，使用 np.linalg.norm 函數來計算 linearity 清單減去每種模式的差的歐氏距離
                overlay = images[np.argmin(distance)]   #根據最小的差異選擇了一個手勢模式對應的圖像，通過np.argmin(distance)找到distance列表中最小值的索引，然後使用該索引選擇相應的圖像，並將其存儲在overlay中

                gestures = [scissor_image, stone_image, paper_image] # 電腦要出的手勢存在變數gestures中
                computer_choice = random.choice(gestures)  # 從名為 gestures 的列表中隨機選擇一個手勢圖像，並將其存儲在 computer_choice 變數中

                if linearity == [0,0,0,0,0]:    #如果手視為石頭，則...
                    glegr = 'stone' #手勢標籤更改為'石頭'
                elif linearity == [0,1,1,0,0]:  #如果手視為剪刀，則...
                    glegr = 'scissor'   #手勢標籤更改為'剪刀'
                elif linearity == [1,1,1,1,1]:  #如果手視為布，則...
                    glegr = 'paper' #手勢標籤更改為'布'
                cv2.putText(image, str(glegr), (cx+100, cy-70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                # 將手勢標籤以文本的形式顯示在影像上，使用 cv2.putText 函數，其中 str(glegr) 是要顯示的文本，(cx+100, cy-70) 是文本的位置，cv2.FONT_HERSHEY_SIMPLEX 是字體，1 是文字的大小縮放因子，(0, 0, 255) 是黃色的BGR顏色碼，1 是文字的線寬，cv2.LINE_AA 是用於抗鋸齒的樣式

                #以上為偵測到的手勢視覺化和處理分析的部分
                #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                #計算如何將一個圖像（在這種情況下是overlay）置中放置在另一個圖像的中心位置
                h, w, _ = image.shape   #獲取image圖像的形狀，並將其分配給h（高度）和w（寬度），是一個三元組，包含圖像的高度、寬度和通道數，在這裡，通道數的信息並没有被使用，因此使用下划線 _ 來表示它
                center_y, center_x = h // 2, w // 2 #計算了image圖像的中心位置，將其分別存儲在center_y（垂直中心）和center_x（水平中心）變數中
                overlay_h, overlay_w, _ = overlay.shape #獲取overlay圖像的形狀，並將其分配給overlay_h（overlay高度）和overlay_w（overlay寬度）。同樣是一個三元組，包含圖像的高度、寬度和通道數
                #計算overlay圖像的起始位置（左上角的坐標），以便將其置中放置在image圖像上。使用了中心位置和overlay圖像的大小來計算起始位置，確保overlay圖像居中放置
                startY = center_y - (overlay_h // 2)
                startX = center_x - (overlay_w // 2)

                #確保overlay圖像不會超出computer_result圖像的邊界，以防止裁切或顯示在畫面之外
                if startY < 0:  #檢查 overlay 圖像的上邊界是否超出了 computer_result 圖像的上邊界。如果 overlay 圖像的上邊界在圖像的上方（即 startY 為負值），則執行以下操作
                    overlay = overlay[-startY:] #裁剪了 overlay 圖像的頂部，刪除了超出圖像上邊界的部分
                    startY = 0  #將 startY 設置為0，以確保 overlay 圖像不再超出圖像的上邊界
                if startX < 0:  #檢查 overlay 圖像的左邊界是否超出了 computer_result 圖像的左邊界。如果 overlay 圖像的左邊界在圖像的左邊（即 startX 為負值），則執行以下操作
                    overlay = overlay[:, -startX:]  #裁剪了 overlay 圖像的左側，刪除了超出圖像左邊界的部分
                    startX = 0  #將 startX 設置為0，以確保 overlay 圖像不再超出圖像的左邊界

                #計算了overlay圖像的底部和右邊的邊界位置。endY是startY與overlay圖像的高度相加，而endX是startX與overlay圖像的寬度相加。
                endY = startY + overlay.shape[0]
                endX = startX + overlay.shape[1]

                if endY > h:    #檢查 overlay 圖像的底部是否超出了 computer_result 圖像的底部。如果 overlay 圖像的底部在圖像的底部之下，則執行以下操作
                    overlay = overlay[:-(endY - h), :]  #裁剪了 overlay 圖像的底部，刪除了超出圖像底部的部分
                    endY = h    #將 endY 設置為圖像的高度 h，以確保 overlay 圖像不再超出圖像的底部
                if endX > w:    #檢查 overlay 圖像的右邊是否超出了 computer_result 圖像的右邊。如果 overlay 圖像的右邊在圖像的右邊之外，則執行以下操作
                    overlay = overlay[:, :-(endX - w)]  #裁剪了 overlay 圖像的右側，刪除了超出圖像右邊的部分
                    endX = w    #將 endX 設置為圖像的寬度 w，以確保 overlay 圖像不再超出圖像的右邊

                # 將 overlay 圖像疊加到 computer_result 圖像上
                computer_result[startY:endY, startX:endX] = overlay #startY 和 endY 定義了垂直方向的放置位置，startX 和 endX 定義了水平方向的放置位置
                #以上為計算如何將影像放置於另一個影像的中心處，並將偵測模式的選擇疊加到"computer_result"上
                #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



                # 在你的主循環中，根據 linearity 的變化來更新 computer_result1
                if prev_linearity is None:
                    prev_linearity = linearity  # 更新 prev_linearity，以便下一幀的比較
                else:
                    if linearity != prev_linearity:
                        if prev_linearity == [0, 0, 0, 0, 0] and linearity != [0, 0, 0, 0, 0]:  # 從石頭變成其他，執行更新操作
                            computer_result1[startY:endY,startX:endX] = computer_choice  # 將 computer_choice 圖像疊加到 computer_result1 圖像上
                        # elif prev_linearity == [0, 0, 0, 0, 0] and linearity == [1, 1, 1, 1, 1]:  # 從石頭變成布，執行更新操作
                        #     computer_result1[startY:endY,startX:endX] = computer_choice  # 將 computer_choice 圖像疊加到 computer_result1 圖像上
                        # elif prev_linearity == [0, 1, 1, 0, 0] and linearity == [0, 0, 0, 0, 0]:  # 從剪刀變成石頭，執行更新操作
                        #     computer_result1[startY:endY,startX:endX] = computer_choice  # 將 computer_choice 圖像疊加到 computer_result1 圖像上
                        elif prev_linearity == [0, 1, 1, 0, 0] and linearity != [0, 1, 1, 0, 0]:  # 從剪刀變成其他，執行更新操作
                            computer_result1[startY:endY,startX:endX] = computer_choice  # 將 computer_choice 圖像疊加到 computer_result1 圖像上
                        elif prev_linearity == [1, 1, 1, 1, 1] and linearity != [1, 1, 1, 1, 1]:  # 從布變成其他，執行更新操作
                            computer_result1[startY:endY,startX:endX] = computer_choice  # 將 computer_choice 圖像疊加到 computer_result1 圖像上
                        # elif prev_linearity == [1, 1, 1, 1, 1] and linearity == [0, 1, 1, 0, 0]:  # 從布變成剪刀，執行更新操作
                        #     computer_result1[startY:endY,startX:endX] = computer_choice  # 將 computer_choice 圖像疊加到 computer_result1 圖像上
                        elif linearity == [1, 1, 0, 0, 1]:    #如果變化成此手勢，則執行以下操作
                            print('遊戲結束')
                            time.sleep (1.5)    #延遲1.5秒
                            exit()  #結束程式
                        elif linearity == [1, 0, 0, 0, 1]:
                            print ('遊戲暫停')
                            time.sleep (1.5)    ##延遲1.5秒
                            os.system("pause")
                        elif linearity == [0, 1, 0, 0, 0]:    #如果變化成此手勢，則執行以下操作
                            print ('切換為偵測模式')
                            computer_mode = "guaranteed_win"  # 模式切換為"偵測模式"
                        elif linearity == [1, 0, 0, 0, 0]:    #如果變化成此手勢，則執行以下操作
                            print('切換為隨機模式')
                            computer_mode = "random"  # 模式切換為"隨機模式"
                        elif glegr == 'paper' and np.array_equal(computer_choice, stone_image): #如果手勢為"布"，且電腦出拳為"石頭"則執行以下操作
                            win_image_resized = cv2.resize(win_image, (endX - startX, endY - startY))   #將win_image縮放大小存到變數win_image_resized中
                            computer_result2[startY:endY, startX:endX] = win_image_resized  #將win_image_resized疊加到computer_result2上
                        elif glegr == 'paper' and np.array_equal(computer_choice, scissor_image):   #如果手勢為"布"，且電腦出拳為"剪刀"則執行以下操作
                            lose_image1_resized = cv2.resize(lose_image1, (endX - startX, endY - startY))   #將lose_image1縮放大小存到變數lose_image1_resized中
                            computer_result2[startY:endY, startX:endX] = lose_image1_resized    #將lose_image1_resized疊加到computer_result2上
                        elif glegr == 'paper' and np.array_equal(computer_choice, paper_image): #如果手勢為"布"，且電腦出拳為"布"則執行以下操作
                            tie_image_resized = cv2.resize(tie_image, (endX - startX, endY - startY))   #將tie_image縮放大小存到變數tie_image_resized中
                            computer_result2[startY:endY, startX:endX] = tie_image_resized  #將tie_image_resized疊加到computer_result2上
                        elif glegr == 'scissor' and np.array_equal(computer_choice, paper_image): #如果手勢為"剪刀"，且電腦出拳為"布"則執行以下操作
                            win_image_resized = cv2.resize(win_image, (endX - startX, endY - startY))   #將win_image縮放大小存到變數win_image_resized中
                            computer_result2[startY:endY, startX:endX] = win_image_resized  #將win_image_resized疊加到computer_result2上
                        elif glegr == 'scissor' and np.array_equal(computer_choice, stone_image): #如果手勢為"剪刀"，且電腦出拳為"石頭"則執行以下操作
                            lose_image1_resized = cv2.resize(lose_image1, (endX - startX, endY - startY))    #將lose_image1縮放大小存到變數lose_image1_resized中
                            computer_result2[startY:endY, startX:endX] = lose_image1_resized    #將lose_image1_resized疊加到computer_result2上
                        elif glegr == 'scissor' and np.array_equal(computer_choice, scissor_image):   #如果手勢為"剪刀"，且電腦出拳為"剪刀"則執行以下操作
                            tie_image_resized = cv2.resize(tie_image, (endX - startX, endY - startY))   #將tie_image縮放大小存到變數tie_image_resized中
                            computer_result2[startY:endY, startX:endX] = tie_image_resized  #將tie_image_resized疊加到computer_result2上
                        elif glegr == 'stone' and np.array_equal(computer_choice, scissor_image):   #如果手勢為"石頭"，且電腦出拳為"剪刀"則執行以下操作
                            win_image_resized = cv2.resize(win_image, (endX - startX, endY - startY))   #將win_image縮放大小存到變數win_image_resized中
                            computer_result2[startY:endY, startX:endX] = win_image_resized  #將win_image_resized疊加到computer_result2上
                        elif glegr == 'stone' and np.array_equal(computer_choice, paper_image): #如果手勢為"石頭"，且電腦出拳為"布"則執行以下操作
                            lose_image1_resized = cv2.resize(lose_image1, (endX - startX, endY - startY))   #將lose_image1縮放大小存到變數lose_image1_resized中
                            computer_result2[startY:endY, startX:endX] = lose_image1_resized    #將lose_image1_resized疊加到computer_result2上
                        elif glegr == 'stone' and np.array_equal(computer_choice, stone_image): #如果手勢為"石頭"，且電腦出拳為"石頭"則執行以下操作
                            tie_image_resized = cv2.resize(tie_image, (endX - startX, endY - startY))   #將tie_image縮放大小存到變數tie_image_resized中
                            computer_result2[startY:endY, startX:endX] = tie_image_resized  #將tie_image_resized疊加到computer_result2上
                        #elif linearity == [0, 1, 1, 0, 0]:
                        #     text1 = 'scissor'
                        #     rect = pygame.draw.rect(win, (255, 0, 0), (500, 720, 300, 70))
                        #     text_surface2 = font.render(text1, True, (255, 0, 0))  # 使用字體對象font來渲染文字字串text，並將渲染後的文字存儲在text_surface1變數中
                        #
                        #     win.blit(text_surface2,(500, 700))  # 將 text_surface1 繪製到 Pygame 視窗 (win) 上，指定了 Surface 應該放置的位置，
                        #     pygame.display.update()  # 更新 Pygame 視窗，以便顯示最新繪製的圖像
                        print("prev_linearity:", prev_linearity)
                        print("linearity:", linearity)
                    prev_linearity = linearity  # 更新 prev_linearity，以便下一幀的比較
                    #computer_result1 = cv2.flip(computer_result1,1)  # cv2.flip(computer_result1, 1)：用於對影像進行水平翻轉，數字1表示水平翻轉，0表示垂直翻轉，-1 表示同時水平和垂直翻轉

                computer_result = cv2.flip(computer_result, 1)  # cv2.flip(computer_result, 1)：用於對影像進行水平翻轉，數字1表示水平翻轉，0表示垂直翻轉，-1 表示同時水平和垂直翻轉

                # c = cv2.waitKey(1)  #等待用戶按下鍵盤上的按鍵，cv2.waitKey(1) 會等待1毫秒，然後返回按下的按鍵的ASCII碼，如果在等待期間未按下任何按鍵，它將返回-1
                # if c == ord('0'):   #檢查用戶是否按下 '0' 鍵。如果是，則執行以下操作
                #     computer_mode = "guaranteed_win"    #模式切換為"偵測模式"
                # elif c == ord('1'): #檢查用戶是否按下 '1' 鍵。如果是，則執行以下操作
                #     computer_mode = "random"    ##模式切換為"隨機模式"
                #
                #
                #
                # if c == ord('2'):   #檢查用戶是否按下 '2' 鍵。如果是，則執行以下操作
                #     computer_choice = random.choice(gestures)   #從名為 gestures 的列表中隨機選擇一個手勢圖像，並將其存儲在 computer_choice 變數中
                #     computer_result1[startY:endY, startX:endX] = computer_choice    # 將 computer_choice 圖像疊加到 computer_result1 圖像上
                #根據按鍵切換模式，並將隨機模式的選擇疊加到"computer_result1"上
                #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # 在 'MediaPipe Hands' 窗口中顯示手部檢測結果
        cv2.imshow('MediaPipe Hands', image)

        image = cv2.cvtColor(cv2.flip(image, 1),cv2.COLOR_BGR2RGB)  # cv2.flip(image, 1)：用於對影像進行水平翻轉，數字1表示水平翻轉，0表示垂直翻轉，-1 表示同時水平和垂直翻轉
        # cv2.cvtColor(...)：用於將影像的顏色空間轉換，在這裡，它將影像從BGR（藍綠紅）色彩空間轉換為RGB（紅綠藍）色彩空間。

        pygame_frame = pygame.surfarray.make_surface(np.rot90(image))   #將OpenCV處理後的圖像轉換為Pygame的Surface對象。首先，使用np.rot90函數將圖像逆時針旋轉90度，以使圖像方向正確（因為攝像頭通常以橫向方向捕捉圖像）。
        # 然後，使用pygame.surfarray.make_surface函數將NumPy數組轉換為Pygame Surface，以便能夠在Pygame窗口中顯示。

        win.blit(pygame_frame, (1060, 0))  # 將 pygame_frame 繪製到 Pygame 視窗 (win) 上，指定了 Surface 應該放置的位置，
        pygame.display.update() #更新 Pygame 視窗，以便顯示最新繪製的圖像


        if (computer_mode == "guaranteed_win"): #檢查 computer_mode 是否等於 "guaranteed_win"，如果是，則執行以下操作
            # 在 'result' 窗口中顯示"偵測模式"計算機結果

            cv2.putText(computer_result, str(computer_mode), (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,cv2.LINE_AA)
            # 將模式名字以文本的形式顯示在影像上，使用 cv2.putText 函數，其中 str(computer_mode) 是要顯示的文本，(50, 120) 是文本的位置，cv2.FONT_HERSHEY_SIMPLEX 是字體，1 是文字的大小縮放因子，(255, 0, 0) 是紅色的BGR顏色碼，1 是文字的線寬，cv2.LINE_AA 是用於抗鋸齒的樣式

            cv2.imshow('result', computer_result)   #顯示result視窗，內容為computer_result

            computer_result = cv2.cvtColor(computer_result, cv2.COLOR_BGR2RGB)  # cv2.cvtColor(...)：用於將影像的顏色空間轉換，在這裡，它將影像從BGR（藍綠紅）色彩空間轉換為RGB（紅綠藍）色彩空間。
            #computer_result = cv2.flip(computer_result, 1)  # 1表示水平翻转
            pygame_frame0 = pygame.surfarray.make_surface(np.rot90(computer_result))    #將OpenCV處理後的圖像轉換為Pygame的Surface對象。首先，使用np.rot90函數將圖像逆時針旋轉90度，以使圖像方向正確（因為攝像頭通常以橫向方向捕捉圖像）。
            # 然後，使用pygame.surfarray.make_surface函數將NumPy數組轉換為Pygame Surface，以便能夠在Pygame窗口中顯示。
            win.blit(pygame_frame0, (10, 0))  # 將 pygame_frame 繪製到 Pygame 視窗 (win) 上，指定了 Surface 應該放置的位置，
            text = 'LETS PLAY A GAME'
            # 渲染文本
            text_surface = font.render(text, True, (255, 0, 0)) #使用字體對象font來渲染文字字串text，並將渲染後的文字存儲在text_surface變數中
            #text：要渲染的文字字串;True：表示是否啟用文字的抗鋸齒（antialiasing）效果。True表示啟用抗鋸齒，可以使文字看起來更加平滑;(255, 0, 0)：表示文字的顏色被設置為紅色，RGB顏色代碼為(255, 0, 0)。
            win.blit(text_surface, (500, 500))   # 將 text_surface 繪製到 Pygame 視窗 (win) 上，指定了 Surface 應該放置的位置，

            lose_image1_resized = cv2.resize(lose_image1, (endX - startX, endY - startY))   #將lose_image1縮放大小存到變數lose_image1_resized中
            computer_result2[startY:endY, startX:endX] = lose_image1_resized    #將lose_image1_resized疊加到computer_result2上
            cv2.imshow('ENDING', computer_result2)  #顯示ENDING視窗，內容為computer_result2

            computer_result2 = cv2.cvtColor(computer_result2,cv2.COLOR_BGR2RGB)  # cv2.cvtColor(...)：用於將影像的顏色空間轉換，在這裡，它將影像從BGR（藍綠紅）色彩空間轉換為RGB（紅綠藍）色彩空間。
            # computer_result2 = cv2.flip(computer_result2, 1)  # 1表示水平翻转
            pygame_frame2 = pygame.surfarray.make_surface(np.rot90(computer_result2))  # 將OpenCV處理後的圖像轉換為Pygame的Surface對象。首先，使用np.rot90函數將圖像逆時針旋轉90度，以使圖像方向正確（因為攝像頭通常以橫向方向捕捉圖像）。
            # 然後，使用pygame.surfarray.make_surface函數將NumPy數組轉換為Pygame Surface，以便能夠在Pygame窗口中顯示。

            win.blit(pygame_frame2, (500, 600))  # 將 pygame_frame1 繪製到 Pygame 視窗 (win) 上，指定了 Surface 應該放置的位置，

            pygame.display.update() #更新 Pygame 視窗，以便顯示最新繪製的圖像
        elif (computer_mode == "random"):   #檢查 computer_mode 是否等於 "random"，如果是，則執行以下操作
            # 在 'result' 窗口中顯示"隨機模式"計算機結果
            cv2.imshow('result', computer_result1)  #顯示result視窗，內容為computer_result1

            cv2.putText(computer_result1, str(computer_mode), (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1,cv2.LINE_AA)
            # 將模式名字以文本的形式顯示在影像上，使用 cv2.putText 函數，其中 str(computer_mode) 是要顯示的文本，(100, 120) 是文本的位置，cv2.FONT_HERSHEY_SIMPLEX 是字體，1 是文字的大小縮放因子，(0, 0, 255) 是紅色的BGR顏色碼，1 是文字的線寬，cv2.LINE_AA 是用於抗鋸齒的樣式

            computer_result1 = cv2.cvtColor(computer_result1, cv2.COLOR_BGR2RGB)     # cv2.cvtColor(...)：用於將影像的顏色空間轉換，在這裡，它將影像從BGR（藍綠紅）色彩空間轉換為RGB（紅綠藍）色彩空間。
            #computer_result1 = cv2.flip(computer_result1, 1)  # 1表示水平翻转
            pygame_frame1 = pygame.surfarray.make_surface(np.rot90(computer_result1))   #將OpenCV處理後的圖像轉換為Pygame的Surface對象。首先，使用np.rot90函數將圖像逆時針旋轉90度，以使圖像方向正確（因為攝像頭通常以橫向方向捕捉圖像）。
            # 然後，使用pygame.surfarray.make_surface函數將NumPy數組轉換為Pygame Surface，以便能夠在Pygame窗口中顯示。


            win.blit(pygame_frame1, (10, 0))  # 將 pygame_frame1 繪製到 Pygame 視窗 (win) 上，指定了 Surface 應該放置的位置，
            text = 'LETS PLAY A GAME'

            # 渲染文本
            text_surface1 = font.render(text, True, (255, 0, 0)) #使用字體對象font來渲染文字字串text，並將渲染後的文字存儲在text_surface1變數中
            # text：要渲染的文字字串;True：表示是否啟用文字的抗鋸齒（antialiasing）效果。True表示啟用抗鋸齒，可以使文字看起來更加平滑;(255, 0, 0)：表示文字的顏色被設置為紅色，RGB顏色代碼為(255, 0, 0)。
            win.blit(text_surface1, (500, 500))   # 將 text_surface1 繪製到 Pygame 視窗 (win) 上，指定了 Surface 應該放置的位置，

            # text：要渲染的文字字串;True：表示是否啟用文字的抗鋸齒（antialiasing）效果。True表示啟用抗鋸齒，可以使文字看起來更加平滑;(255, 0, 0)：表示文字的顏色被設置為紅色，RGB顏色代碼為(255, 0, 0)。


            cv2.imshow('ENDING', computer_result2)  #顯示ENDING視窗，內容為computer_result2

            computer_result2 = cv2.cvtColor(computer_result2,cv2.COLOR_BGR2RGB)  # cv2.cvtColor(...)：用於將影像的顏色空間轉換，在這裡，它將影像從BGR（藍綠紅）色彩空間轉換為RGB（紅綠藍）色彩空間。
            #computer_result2 = cv2.flip(computer_result2, 1)  # 1表示水平翻转
            pygame_frame2 = pygame.surfarray.make_surface(np.rot90(computer_result2))  # 將OpenCV處理後的圖像轉換為Pygame的Surface對象。首先，使用np.rot90函數將圖像逆時針旋轉90度，以使圖像方向正確（因為攝像頭通常以橫向方向捕捉圖像）。
            # 然後，使用pygame.surfarray.make_surface函數將NumPy數組轉換為Pygame Surface，以便能夠在Pygame窗口中顯示。

            win.blit(pygame_frame2, (500, 600))  # 將 pygame_frame1 繪製到 Pygame 視窗 (win) 上，指定了 Surface 應該放置的位置，

            pygame.display.update() #更新 Pygame 視窗，以便顯示最新繪製的圖像

        c = cv2.waitKey(1)
        if c == 27: #檢查 c 是否等於27，即按下的鍵是否為 'Esc' 鍵的ASCII碼。如果是，則執行以下操作
            break   #跳出 while 迴圈，結束主要的影像捕獲和處理迴圈


cap.release()   #釋放了攝像頭資源，關閉了攝像頭
cv2.destroyAllWindows() #關閉了所有OpenCV的視窗，這樣在程式結束時不會留下任何未關閉的窗口
#根據不同模式顯示不同的結果，並在按下"Esc"鍵後，跳出迴圈，結束一切
