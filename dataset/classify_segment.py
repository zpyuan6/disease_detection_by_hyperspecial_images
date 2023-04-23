import os
import shutil

def copy_background_segment(source_path, target_path):
    background_path = os.path.join(target_path,"background")
    tree_path = os.path.join(target_path,"tree")

    tree_list = []

    for root,folders, files in os.walk(tree_path):
        tree_list.extend(files)
    
    for root,folders, files in os.walk(source_path):
        for file in files:
            if str(file).find("superpixel")!=-1 and (not file in tree_list):
                shutil.copyfile(os.path.join(root,file), os.path.join(background_path,file))

if __name__ == "__main__":
    source_path = "F:\\Hyperspecial\\pear\\15_07_22\\segment"
    target_path = "F:\\Hyperspecial\\pear_processed\\classifier_training_data"
    copy_background_segment(source_path, target_path)