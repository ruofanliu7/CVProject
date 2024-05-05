import scipy.io as sio
import cv2

f = open("example.txt", "r")
data = [element.strip('][').split(', ') for element in f.read().split("\n")]
frames = [int(d[0]) for d in data]
y_positions = [int(d[1]) for d in data]

ball_color = (255, 0, 0)
knee_color = (0, 255, 0)
thickness = 3 

over_line = True
juggling_count = 0
frame_img = cv2.imread(f"detected_locations/juggling3/frame{0}_detected.jpg")
print(frame_img.shape)
e
video=cv2.VideoWriter('juggling2_offline.mp4',0,24,(frame_img.shape[1], frame_img.shape[0]))


for frame_num in range(240):
    frame_dat = sio.loadmat(f"detected_locations/juggling3/frame{frame_num}.mat")
    frame_img = cv2.imread(f"detected_locations/juggling3/frame{frame_num}_detected.jpg")
    knee_height = int(frame_dat['9'][0][1]) if '9' in frame_dat else knee_height
    lined_img = cv2.line(frame_img, (0, knee_height), (frame_img.shape[1] - 2, knee_height), knee_color, thickness)

    if frame_num in frames:
        i = frames.index(frame_num)
        frame = frames[i]
        y_pos = y_positions[i]

        lined_img = cv2.line(lined_img, (0, y_pos), (frame_img.shape[1] - 2, y_pos), ball_color, thickness)
            
        if not over_line and y_pos < knee_height:
            juggling_count += 1

        over_line = y_pos < knee_height

    text = f"Count: {juggling_count}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = 220  # Adjust as needed
    text_y = 50  # Adjust as needed
    cv2.putText(lined_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    video.write(lined_img)
cv2.destroyAllWindows()
video.release()