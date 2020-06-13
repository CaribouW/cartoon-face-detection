import config
# 根据train.txt 生成 适用于opencv 的文本
if __name__ == '__main__':
    output_path = 'dataset/cartoon_dataset/train_bbox_gt.txt'
    with open('dataset/cartoon_dataset/train.txt', 'r') as f:
        annotations = f.readlines()
    n_lines = len(annotations)
    idx_line = 0
    img_idx = 1
    while idx_line < n_lines:
        line = annotations[idx_line].strip()
        img_name = line.split(',')[0]
        cur_img_idx = int(line.split('.')[0])
        poses = []
        while cur_img_idx == img_idx:
            x_min, y_min, x_max, y_max = [int(x) for x in annotations[idx_line].split(',')[1:]]
            w, h = x_max - x_min, y_max - y_min
            pos_line = ' '.join([str(x_min), str(y_min), str(w), str(h)])
            idx_line += 1
            poses.append(pos_line)
            if idx_line < n_lines:
                cur_img_idx = int(annotations[idx_line].strip().split('.')[0])
            else:
                break

        contents = [img_name, str(len(poses))] + poses

        with open(output_path, 'a') as out:
            content_str = '\n'.join(contents)
            out.write(content_str + '\n')
        img_idx += 1
    pass
