text = ["no lemon,no melon", "kayak", "was it a car or a cat i saw"]

cleanwords = []

for item in text:
    if " " in item:
        cleanwords.append("".join(item.split()))
    else:
        cleanwords.append(item)

print(cleanwords)
