# count = 0
# with open("val.txt", 'r') as f:
#     datas = f.readlines()
#     for data in datas:
#         _ , label = data.split("#")
#         if int(label) == 1:
#             count += 1
#
# print(count)

grid_anchors = [[[None for i in range(1)]for j in range(1)]for k in range(2)]
grid_anchors[1][0][0] = 1
print(grid_anchors)