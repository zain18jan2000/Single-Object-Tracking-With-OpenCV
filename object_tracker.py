import cv2

trackers = {
    'csrt' : cv2.legacy.TrackerCSRT_create,  # hight accuracy ,slow
    'mosse' : cv2.legacy.TrackerMOSSE_create,  # fast, low accuracy
    'kcf' : cv2.legacy.TrackerKCF_create,   # moderate accuracy and speed
    'medianflow' : cv2.legacy.TrackerMedianFlow_create,
    'mil' : cv2.legacy.TrackerMIL_create,
    'tld' : cv2.legacy.TrackerTLD_create,
    'boosting' : cv2.legacy.TrackerBoosting_create
}

tracker_key = 'kcf'
roi = None
tracker = trackers[tracker_key]()

cap = cv2.VideoCapture('1.mp4')

while True:
    frame = cap.read()[1]

    if frame is None:
        break

    frame = cv2.resize(frame,(750,550))

    if roi is not None:
        success, box = tracker.update(frame)

        if success:
            x,y,w,h = [int(c) for c in box]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255,0,0),2)
        else:
            print('Tracking failed')

            roi = None
            tracker = trackers[tracker_key]()

    cv2.imshow('Tracking',frame)
    k = cv2.waitKey(30)

    if k == ord('s'):
        roi = cv2.selectROI('Tracking',frame)

        tracker.init(frame,roi)

    elif k == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()