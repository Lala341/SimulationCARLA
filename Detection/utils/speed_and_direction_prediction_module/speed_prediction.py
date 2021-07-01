from utils.image_utils import image_saver

is_vehicle_detected = [0]
current_frame_number_list = [0]
current_frame_number_list_2 = [0]
bottom_position_of_detected_vehicle = [0]


def predict_speed(
    top,
    bottom,
    right,
    left,
    current_frame_number,
    crop_img,
    roi_position,
    ):
    speed = 'n.a.'  # means not available, it is just initialization
    direction = 'n.a.'  # means not available, it is just initialization
    update_csv = False
    is_vehicle_detected.insert(0, 1)
    update_csv = True
    image_saver.save_image(crop_img)  # save detected vehicle image
    current_frame_number_list_2.insert(0, current_frame_number)
    # for debugging
    # print("bottom_position_of_detected_vehicle[0]: " + str(bottom_position_of_detected_vehicle[0]))
    
    return (direction, speed, is_vehicle_detected, update_csv)
