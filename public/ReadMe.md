seq/ (e.g. seq1, seq2, seq3)
    dataset/
        {time_stamp}/ (e.g. 1681710717_532211005)
            1. camera.csv: 
                camera name
            2. detect_road_marker.csv:
                a. detected bounding boxes, the bounding box are not always correct.
                b. format: (x1, y1, x2, y2, class_id, probability)
                c. class_id: (0:zebracross, 1:stopline, 2:arrow, 3:junctionbox, 4:other)
            3. initial_pose.csv:
                initial pose for ICP in "base_link" frame.
            4. raw_image.png:
                captured RGB image
            5. sub_map.csv:
                map points for ICP, (x, y, z).
            6. gound_turth_pose.csv: """not exist in all dirs"""
                x, y localization ground turth in "base_link" frame.

    other_data/
        {timestamp}_raw_speed.csv: (e.g. 1681710717_572170877_raw_speed.csv)
            car speed(km/hr)
        {timestamp}_raw_imu.csv:
            1st line: orientation: x, y, z, w
            2nd line: angular_velocity: x, y, z
            3rd line: linear_acceleration: x, y, z

    all_timestamp.txt:
        list all directories in time order
    localization_timestamp.txt:
        list all directories with "gound_turth_pose.csv" in time order


camera_info/
    {camera}/ (e.g. lucid_cameras_x00)
        {camera_name}_camera_info.yaml: (e.g. gige_100_b_hdr_camera_info.yaml)
            intrinsic parameters
        {camera_name}_mask.png:
            The mask for the ego car show in the image, it could help for decreasing some false alarms in detection.
        camera_extrinsic_static_tf.launch:
            transformation parameters between cameras
            key_word: tf2_ros, Robot Operating System (ROS)

Hint:
    1. Use "base_link" as the main frame
    2. you could align the map and your detection in bird's eye view (top-down view), which means don't need to regard the z value.
