def read_file(path):
    text_file = open(path, "r")
    test = open(path, "r").read().split('\n')
    # print(test)
    # lines = text_file.readlines()
    # print(lines)
    # print(len(lines))
    text_file.close()
    return test


print('asfasdf')
the_words = read_file('Positive.txt')
print(the_words)