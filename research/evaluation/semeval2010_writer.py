def file_writer(classes, dirs, ids, file_, inv_map, class_type="true"):
    length = len(dirs)
    for count in range(length):
        if class_type == "true":
            value = classes[count].item()
            dir_ = dirs[count].item()
        else:
            value = classes[count].max(0)[1].item()
            dir_ = dirs[count].max(0)[1].item()
        txt = str(ids[count].item()) + "\t" + inv_map[value]
        if inv_map[value] != "Other":
            if dir_ == 0:
                txt += "(e1,e2)"
            else:
                txt += "(e2,e1)"
        file_.write(txt + "\n")
    return file_
