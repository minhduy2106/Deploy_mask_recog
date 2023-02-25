from prepare import *

# Face recognition by image
def pipeline_model(path, filename):
    # %%time
    path_query = path
    orgimg = cv2.imread(path_query)  # BGR
    img = resize_image(orgimg.copy(), size_convert, orgimg)

    with torch.no_grad():
        pred = model_detect_face(img[None, :])[0]

    # Apply NMS
    det = non_max_suppression(pred, conf_thres, iou_thres)[0]
    bboxs = np.int32(scale_coords(
        img.shape[1:], det[:, :4], orgimg.shape).round().cpu().numpy())

    for i in range(len(bboxs)):
        x1, y1, x2, y2 = bboxs[i]
        rois = orgimg[y1:y2, x1:x2]

        final_img = cv2.resize(rois, (224, 224))
        final_img = cv2.normalize(final_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8U)
        final_img = final_img.reshape((-1, 224, 224, 3))
        Predict = model_recog_mask.predict(final_img)
        Predict = Predict[0, 0]
        print(Predict)

        if (Predict < 0.2):
            status = f'No Mask! Please wear mask {round(Predict * 100)}%'
            color = (0, 0, 255)
            cv2.rectangle(orgimg, (x1, y1), (x2, y2), color, 3)
            cv2.putText(orgimg, status, (x1, y1 - 20), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        else:
            status = f"Masked {round(Predict * 100)}%"
            color = (0, 255, 0)
            cv2.rectangle(orgimg, (x1, y1), (x2, y2), color, 3)
            cv2.putText(orgimg, status, (x1, y1 - 20), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    cv2.imwrite('./static/predict/{}'.format(filename), orgimg)


def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            start_time = time.time()
            img = resize_image(frame.copy(), size_convert, frame)
            with torch.no_grad():
                pred = model_detect_face(img[None, :])[0]

            det = non_max_suppression(pred, conf_thres, iou_thres)[0]
            bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], frame.shape).round().cpu().numpy())

            if len(bboxs) <= 0:
                continue

            for i in range(len(bboxs)):
                x1, y1, x2, y2 = bboxs[i]
                rois = frame[y1:y2, x1:x2]

                final_img = cv2.resize(rois, (224, 224))
                final_img = cv2.normalize(final_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                          dtype=cv2.CV_8U)
                final_img = final_img.reshape((-1, 224, 224, 3))
                Predict = model_recog_mask.predict(final_img)
                Predict = Predict[0, 0]
                print(Predict)

                if (Predict < 0.2):
                    status = f'No Mask! Please wear mask {round(Predict * 100)}%'
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, status, (x1, y1 - 20), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                else:
                    status = f"Masked {round(Predict * 100)}%"
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, status, (x1, y1 - 20), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            # calculate fps
            elapsed_time = time.time() - start_time
            fps = 1 / elapsed_time
            fps_str = f'FPS: {fps:.1f}'
            cv2.putText(frame, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')