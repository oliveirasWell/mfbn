def spark_debug_print(collected_list):
    for item in collected_list:
        print('--------------------------------------------------------')
        print('/--------------------------------------------------------/')
        print(item[0])
        item_list = item[1]
        for i_of_list in item_list:
            print(i_of_list)
        print('/--------------------------------------------------------/')
        print('--------------------------------------------------------')