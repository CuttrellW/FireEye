import cv2
import os
import tkinter
from datetime import datetime
from image_classifier import predict_data
from ctypes import *


def begin_watch(model, watch_config):
    scan_frequency = watch_config['frequency'] * 30  # fps
    alarm_threshold = watch_config['confidenceThreshold']
    delete_trigger_frame = watch_config['deleteTriggerFrame']
    feed = cv2.VideoCapture(0)

    frame_count = 1
    alarm_triggered = False

    while not alarm_triggered:
        ret, frame = feed.read()
        if not ret:
            print("Unable to retrieve frames from feed")
            break
        cv2.imshow("FireEye", frame)

        key = cv2.waitKey(1)
        if key % 256 == 27:
            # ESC key
            print("Ending watch")
            break
        elif frame_count % scan_frequency == 0:
            cv2.imwrite("sample_frame.png", frame)
            print("Sample captured")
            prediction = predict_data(model)
            if prediction['class'] == 'fire' and prediction['score'] > alarm_threshold:
                alarm_triggered = True
                print("Alarm triggered!")

        frame_count += 1

    feed.release()

    cv2.destroyAllWindows()

    if delete_trigger_frame:
        os.remove("sample_frame.png")

    return alarm_triggered


def alert(config):
    now = datetime.now()
    date_time = now.strftime("%H:%M:%S, %m/%d/%Y")
    message = f"WARNING: A FIRE HAS BEEN DETECTED IN THE AREA.\nTime of detection: {date_time}"

    if config['style'] == 0:
        print(message)
    elif config['style'] == 1:
        tk = tkinter.Tk()
        tk.title("FireEye Alert")
        tkinter.Label(tk, text=message).grid(column=0, row=0, padx=20, pady=30)
        tk.mainloop()
