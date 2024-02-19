# def replace_list_values(lst):
#     # Encuentra el valor máximo en la lista
#     max_val = max(lst)

#     # Recorre la lista y cambia el valor máximo por 1 y todos los demás valores por 0
#     for i in range(len(lst)):
#         lst[i] = 1 if lst[i] == max_val else 0

#     return lst

# lst1 = [3.8933172e-09, 2.6332248e-07 , 9.9999976e-01]
# lst2 = [2.6332248e-07 , 9.9999976e-01,3.8933172e-09]
# lst3 = [9.9999976e-01, 2.6332248e-07 ,3.8933172e-09]
# print(replace_list_values(lst1))  # [0, 0, 1]
# print(replace_list_values(lst2))  # [0, 1, 0]
# print(replace_list_values(lst3))  # [1, 0, 0]

# #Predicción: [[3.8933172e-09 2.6332248e-07 9.9999976e-01]]




array = [[5, 10, 15], [20, 25, 30], [35, 40, 45]]

# Para cada sublista en la lista
for sublst in array:
    # Encuentra el valor máximo en la sublista
    max_val = max(sublst)

    # Recorre la sublista y cambia el valor máximo por 1 y todos los demás valores por 0
    for i in range(len(sublst)):
        sublst[i] = 1 if sublst[i] == max_val else 0

print(array)  # [[0, 0, 1], [0, 0, 1], [0, 0, 1]]





# array = [[5, 100, 15]]

# # Accede a la única sublista en la lista
# sublst = array[0]

# # Encuentra el valor máximo en la sublista
# max_val = max(sublst)

# # Recorre la sublista y cambia el valor máximo por 1 y todos los demás valores por 0
# for i in range(len(sublst)):
#     if sublst[i] == max_val:
#         sublst[i] = 1
#         if i == 0:
#             print("cebolla")
#         elif i == 1:
#             print("chile")
#         elif i == 2:
#             print("tomate")
#     else:
#         sublst[i] = 0
    