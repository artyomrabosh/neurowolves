def make_list():
    f = open("cuts/tagged.txt", "r", encoding="utf-8")
    new = open("cuts/wolvescut2.txt", "w", encoding="utf-8")

    txt=[]
    for i in f:
        i=i.replace("<|startoftext|>", "<s>")
        i=i.replace("<|endoftext|>", "</s>")
        new.write(i)



make_list()

first = open("cuts/tagged2.txt", "r", encoding="utf-8")
second = open("cuts/wolvescut2.txt", "r", encoding="utf-8")

phrases = []
for i in first:
    phrases.append(i)
for i in second:
    phrases.append(i)

print(len(phrases))
print(len(list(set(phrases))))

final = open("cuts/wolves_final.txt", "w", encoding="utf-8")

for i in phrases:
    final.write(i)