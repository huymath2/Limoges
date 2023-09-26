import itertools

letters = 'abcdefghijklmnopqrstuvwxyz'

words = [''.join(x) for x in itertools.product(letters, repeat=4)]

print(len(words))
with open('pwd_list.txt', 'w') as file:
    for word in words:
        file.write(''.join(word) + '\n')
