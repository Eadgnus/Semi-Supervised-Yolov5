import cv2
import yaml
import os
import detect_ssl
import tqdm

'''
type: teacher, student (teacher의 pseudo label이면 teacher)

mode: train, valid, test

yaml_path: label dataset yaml path
    
list_label: [class \t bbox, class2 \t bbox2, ...] 공백없이

image_name: image이름

image: image 영상
'''


def save_pseudo_label(type, mode, yaml_path, list_label, image_name, image):
    remove_name = image_name.split(".")[-1]
    image_name = image_name.split(f".{remove_name}")[0]
    pre_path = yaml_path.split(".yaml")[0]

    # E:abc 와같이 드라이브의 바로 하위 폴더일 경우
    if len(pre_path.split("/")[:-2]) == 0:
        list_path = pre_path.split("/")[:-1][0].split(":")[0] + ":\\"
        path = list_path
    else:
        list_path = pre_path.split("/")[:-2]
        path = str(os.path.join(*list_path))

    # 기존 datasets과 같은 위치에 생성
    output_dir = os.path.join(path, "teacher_pseudo_label") if type == "teacher" else os.path.join(path,
                                                                                                   "student_pseudo_label")
    if not os.path.exists(output_dir):
        # output 폴더 만들기
        os.makedirs(output_dir, exist_ok=True)
        # output 폴더의 datasets 폴더 만들기
        for tvt_mode in ["train", "valid", "test"]:
            os.makedirs(os.path.join(output_dir, tvt_mode, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, tvt_mode, "labels"), exist_ok=True)

    output_image_name = os.path.join(output_dir, mode, "images", f"{image_name}.jpg")
    output_label_name = os.path.join(output_dir, mode, "labels", f"{image_name}.txt")
    name_cnt = 0

    # yaml 체크
    #  pseudo_yaml, yaml_file_path, new_train_path, new_valid_path, new_test_path
    new_yaml_path = check_yaml(
        pseudo_yaml=f"{output_dir}/data.yaml",
        yaml_file_path=yaml_path,
        new_train_path=f"{output_dir}/train/images",
        new_valid_path=f"{output_dir}/valid/images",
        new_test_path=f"{output_dir}/test/images",
        type=type
    )
    # 라벨 중복 체크
    # 데이터 양이 걷잡을 수 없이 늘어나 취소
    # while os.path.isfile(f"{output_label_name}"):
    #     name_cnt += 1
    output_image_name = os.path.join(output_dir, mode, "images", f"{image_name}.jpg")
    output_label_name = os.path.join(output_dir, mode, "labels", f"{image_name}.txt")

    with open(output_label_name, "w") as txt:
        for label in list_label:
            txt.write(f"{label[0]}\n")

    cv2.imwrite(output_image_name, image)
    return new_yaml_path


'''
pseudo_yaml: 데이터셋 path

yaml_file: label 데이터셋의 yaml

new_train_path: 새로 추가될 train path

new_valid_path: 새로 추가될 valid path

new_test_path: 새로 추가될 test path
'''


def check_yaml(pseudo_yaml, yaml_file_path, new_train_path, new_valid_path, new_test_path, type):
    new_yaml_path = f"{pseudo_yaml}"
    if not os.path.exists(new_yaml_path):
        with open(yaml_file_path, 'r') as file:
            # 기존 YAML 파일 읽기
            yaml_data = yaml.safe_load(file)

            if type == 'student':
                yaml_txt = f'''train: 
    - {yaml_data['train']}
    - {new_train_path}
val: 
    - {yaml_data['val']}
    - {new_valid_path}
test: 
    - {yaml_data['test']}
    - {new_test_path}
nc: {yaml_data['nc']}
names: {yaml_data['names']}
'''
            else:
                yaml_txt = f'''train: 
    - {new_train_path}
val: 
    - {new_valid_path}
test: 
    - {new_test_path}
nc: {yaml_data['nc']}
names: {yaml_data['names']}
'''
            # 새로운 경로 추가
            # print(yaml_data['train'])
            # yaml_data['train'] = [yaml_data['train'], new_train_path]
            # yaml_data['val'] = [yaml_data['val'], new_valid_path]
            # yaml_data['test'] = [yaml_data['test'], new_test_path]

        # 새로운 YAML 파일로 저장
        with open(pseudo_yaml, 'w') as file:
            file.write(yaml_txt)
    return new_yaml_path


def run(unlabel_path, weights, data, type):
    data = data.replace("\\", "/")
    unlabel_list = os.listdir(unlabel_path)
    mode_count = 0
    new_yaml_path = ""

    for img_name in tqdm.tqdm(unlabel_list):
        img_full_path = os.path.join(unlabel_path, img_name)
        img, label_list = detect_ssl.run(
            weights=weights,
            data=data,
            source=img_full_path,
            device="0",
            exist_ok=True
        )
        # train, val, test를 나누는 기준 정하고 데이터셋 and yaml 생성(mode)
        if mode_count < 7:
            mode = "train"
        elif 7 <= mode_count < 9:
            mode = "valid"
        else:
            mode = "test"
            # 아래에서 +1 해주기 때문에 -1로 하여 0부터 시작
            mode_count = -1
        
        # yaml path 리턴 and 데이터 셋 생성
        new_yaml_path = save_pseudo_label(
            type=type,
            mode=mode,
            yaml_path=data,
            list_label=label_list,
            image_name=img_name,
            image=img
        )
        mode_count += 1
    return new_yaml_path


# unlabel_path = r"E:\ssl_test_datasets\test\images"
# run(
#     unlabel_path=unlabel_path,
#     weights="run/ssl/teacher/weights/best.pt",
#     data=r"E:\ssl_test_datasets/data.yaml",
#     type="teacher"
# )
# test_image = cv2.imread(r"E:\ssl_test_datasets\test\images\image_77_jpg.rf.75925876e87ad1b9972a2951cf758b83.jpg")
# save_pseudo_label("student", "train", r"E:/ssl_test_datasets/data.yaml", ["0\t1 2 3 4", "1\t1 2 3 4", "2\t1 2 3 4"], "test_img.jpg", test_image)
