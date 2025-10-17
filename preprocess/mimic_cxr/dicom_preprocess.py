import os
import json
from tqdm import tqdm

def load_json(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)


def save_json(data, file_path, encoding="utf-8", indent=2):
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=indent)


def preprocess_dicom(json_file, folder_path_dict):
                        
    result = []
    for i in tqdm(json_file):
        dicom_id = i['dicom_id']
        if dicom_id in folder_path_dict:
            i['original_dicom_id'] = folder_path_dict[dicom_id]
            result.append(i)
        else:
            raise ValueError
    assert len(result) == len(json_file)
    
    return result    


def make_folder_path_dict(data_root):
    
    mimic_cxr_dicom_root_dir = os.path.join(data_root, 'MIMIC-CXR-DICOM_JPG/mimic-cxr-2.0.0.physionet.org/files')

    folder_path_dict = {}
    for root, _, files in tqdm(os.walk(mimic_cxr_dicom_root_dir)):
        for file in files:
            if file.endswith('.jpg'):
                    new_dicom_id = root.replace(data_root + '/', '')
                    new_dicom_id = os.path.join(new_dicom_id, file)
                    folder_path_dict[file] = new_dicom_id
                    
    return folder_path_dict

if __name__ == "__main__":
    data_root = 'datasets'
    version = 'v11.0'
    # v10.0 preprocess path
    v10_preprocess_dir = os.path.join(data_root, 'MIMIC-CXR/preprocess/v10.0')
    save_dir = os.path.join(data_root, 'MIMIC-CXR/preprocess', version)
    os.makedirs(save_dir, exist_ok=True)
    
    
    mimic_cxr_dicom_root_dir = os.path.join(data_root, 'MIMIC-CXR-DICOM_JPG/mimic-cxr-2.0.0.physionet.org/files')

    
    folder_path_dict = {}
    for root, _, files in tqdm(os.walk(mimic_cxr_dicom_root_dir)):
        for file in files:
            if file.endswith('.jpg'):
                    new_dicom_id = root.replace(data_root + '/', '')
                    new_dicom_id = os.path.join(new_dicom_id, file)
                    folder_path_dict[file] = new_dicom_id

    split_list = ['train', 'validate', 'test']
    
    folder_path_dict = make_folder_path_dict(data_root)
    
    
    for split in split_list:
        pre_version_file_path = os.path.join(v10_preprocess_dir, f'{split}.json')
        pre_version_file = load_json(pre_version_file_path)
        
        result = preprocess_dicom(pre_version_file, folder_path_dict)
        
        save_path = os.path.join(save_dir, f"{split}.json")
        
        save_json(result, save_path)
        
        
