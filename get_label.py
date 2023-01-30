# import os
# f = open("val1.txt",'w')
# root = "/home/qcj/cvitek/val"
# lst = os.listdir(root)
# print(lst)
# for every_dir in lst:
#     img_father = os.path.join(os.path.join(root,every_dir),'masked')
#     imgs = os.listdir(img_father)
#     for img_path in imgs:
#         f.write(os.path.join(img_father, img_path) + "#1\n")
#
#     img_father = os.path.join(os.path.join(root, every_dir), 'unmasked')
#     imgs = os.listdir(img_father)
#     for img_path in imgs:
#         f.write(os.path.join(img_father, img_path)  + "#0\n")
#
# f.close()
#
with open("va12.txt", 'r') as r:
    with open("val3.txt", 'w') as w:
        line = r.readline()
        while line:
            w.write(line.replace("/home/qcj/cvitek/val", "/work/val"))
            line = r.readline()