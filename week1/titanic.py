"""
Какое количество мужчин и женщин ехало на корабле?
В качестве ответа приведите два числа через пробел.
"""
import pandas
data = pandas.read_csv("titanic.csv", index_col="PassengerId")
sex = data["Sex"]
print("Number of MEN:", sex.value_counts()[0])  # Number of MEN: 577
print("Number of WOMEN:", sex.value_counts()[1])  # Number of WOMEN: 314


"""
Какой части пассажиров удалось выжить?
Посчитайте долю выживших пассажиров. Ответ приведите
в процентах (число в интервале от 0 до 100,
знак процента не нужен).
"""
survived = data["Survived"]
print("Percentage of those who survived:", sum(survived) / len(survived) * 100)  # 38.38


"""
Какую долю пассажиры первого класса составляли среди всех пассажиров?
Ответ приведите в процентах
(число в интервале от 0 до 100, знак процента не нужен).
"""
print(sum(data["Pclass"] == 1) / len(data["Pclass"]) * 100,
      "per cent of the passengers who survived were from 1-st class")  # 24.24


"""
Какого возраста были пассажиры?
Посчитайте среднее и медиану возраста пассажиров.
В качестве ответа приведите два числа через пробел.
"""
mean_age = data.mean()[2]
median_age = data.median()[2]
print("Mean age:", mean_age)  # 29.7
print("Median age", median_age)  # 28


"""
Коррелируют ли число братьев/сестер с числом родителей/детей?
Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
"""
bros = data["SibSp"]
parents = data["Parch"]
print("Correlation between number of brothers/sisters and number of parents/children", bros.corr(parents))  # 0.41


"""
Какое самое популярное женское имя на корабле?
Извлеките из полного имени пассажира (колонка Name)
его личное имя (First Name).
"""

whole_names = data.loc[data['Sex'] == 'female', 'Name']
new = ""
for name in whole_names:
    # Delete noise and save the result in variable "new"
    new_name = name.replace("Miss.", "").replace("Mrs.", "").replace(",", "").replace("(", "").replace(")", "")
    new += new_name

splitted = new.split()
counts = {}
for name in set(splitted):
    counts[name] = splitted.count(name)


print("The most popular names:",
      sorted(counts, key=counts.get, reverse=True)[:3])  # We choose Anna because "William" is a surname
