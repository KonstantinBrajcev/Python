# # ПАЛИНДРОМ---------------------------------------------------
# class Solution(object):
#     def isPalindrome(self, x):
#         """
#         :type x: int
#         :rtype: bool
#         """
#         x = str(x)
#         end = int(round(len(x)/2))
#         is_palindrome = True
#         for i in range(end):
#             if x[i] != x[-i-1]:
#                 is_palindrome = False
#                 break
#             else:
#                 is_palindrome = True
#         return is_palindrome


# x = int(input())
# solution = Solution()
# print(solution.isPalindrome(abs(x)))


# ПЕРЕВОД ГРЕЧЕСКИХ ЦИФР-------------------------------------------
# class Solution:
#     def romanToInt(self, s: str) -> int:
# sum = 0
# for i in range(len(s)):
#     if s[i] == "M":
#         if i == 0:
#             sum += 1000
#         elif s[i-1] != "C":
#             sum += 1000
#         elif s[i-1] == "C":
#             sum += 800
#     if s[i] == "D":
#         if i == 0:
#             sum += 500
#         elif s[i-1] != "C":
#             sum += 500
#         elif s[i-1] == "C":
#             sum += 300
#     if s[i] == "C":
#         if i == 0:
#             sum += 100
#         elif s[i-1] != "X":
#             sum += 100
#         elif s[i-1] == "X":
#             sum += 80
#     if s[i] == "L":
#         if i == 0:
#             sum += 50
#         elif s[i-1] != "X":
#             sum += 50
#         elif s[i-1] == "X":
#             sum += 30
#     if s[i] == "X":
#         if i == 0:
#             sum += 10
#         elif s[i-1] != "I":
#             sum += 10
#         elif s[i-1] == "I":
#             sum += 8
#     if s[i] == "V":
#         if i == 0:
#             sum += 5
#         elif s[i-1] != "I":
#             sum += 5
#         elif s[i-1] == "I":
#             sum += 3
#     if s[i] == "I":
#         sum += 1

# return sum
# ВТОРОЙ ВАРИАНТ
#         roman_values = {
#             "I": 1,
#             "V": 5,
#             "X": 10,
#             "L": 50,
#             "C": 100,
#             "D": 500,
#             "M": 1000
#         }
#         sum = 0
#         prev_value = 0
#         for i in range(len(s)-1, -1, -1):
#             current_value = roman_values[s[i]]
#             if current_value < prev_value:
#                 sum -= current_value
#             else:
#                 sum += current_value
#             prev_value = current_value

#         return sum


# s = input()
# solution = Solution()
# print(solution.romanToInt(s))

# НАЙТИ СУММУ ЧИСЕЛ--------------------------------------------
# class Solution(object):
#     def twoSum(self, nums, target):
#         """
#         :type nums: List[int]
#         :type target: int
#         :rtype: List[int]
#         """
#         # if nums[0] + nums[1] == target:
#         #     return [0, 1]
#         for i in range(len(nums)):
#             for j in range(1, len(nums)):
#                 if (nums[i] + nums[j] == target) and (i != j):
#                     return [i, j]


# nums = [2, 5, 5, 11]
# target = 10
# solution = Solution()
# print(solution.twoSum(nums, target))

# НАЙТИ ОДИНАКОВЫЕ БУКВЫ В СЛОВЕ
# from typing import List

# class Solution:
#     def longestCommonPrefix(self, strs: List[str]) -> str:
#         if not strs:
#             return ""

#         # Находим самое короткое слово в списке
#         shortest_word = min(strs, key=len)

#         # Перебираем буквы самого короткого слова
#         for i, char in enumerate(shortest_word):
#             for z in strs:
#                 if z[i] != char:
#                     return shortest_word[:i]
#         return shortest_word


# strs = ["flower", "flow", "flight"]
# solution = Solution()
# print(solution.longestCommonPrefix(strs))


# ПРОВЕРКА ЗАКРЫТОСТИ СКОБОК
# class Solution:
#     def isValid(self, s: str) -> bool:
#         is_Valid = True
#         for i in range(len(s)):
#             if s[i] == "(" and (s[i+1] != ")" or s.find(")") < 0):
#                 return False
#             if s[i] == "{" and (s[i+1] != "}" or s.find("}") < 0):
#                 return False
#             if s[i] == "[" and (s[i+1] != "]" or s.find("]") < 0):
#                 return False
#         return is_Valid


# s = "{[]}"
# solution = Solution()
# print(solution.isValid(s))

# ВТОРОЙ ВАРИАНТ РЕШЕНИЯ
# class Solution:
#     def isValid(self, s: str) -> bool:
#         stack = []  # Создаем массив последовательности СТЕК
#         mapping = {")": "(", "}": "{", "]": "["}  # Используем словарь
#         for char in s:  # Перебираем символы
#             if char in mapping:  # Проверяем словарь
#                 top_element = stack.pop() if stack else '#'
#                 if mapping[char] != top_element:  # Проверяем
#                     return False
#             else:
#                 stack.append(char)  # Добавляем символ в стэк
#         return not stack  # Проверяем пустой ли стек


# # s = "(){}[]"
# # s = "{[((]))]}"
# s = "(("
# solution = Solution()
# print(solution.isValid(s))


# -------------------Вывод двух списков
# from typing import List
# class Solution:
#     def getConcatenation(self, nums: List[int]) -> List[int]:
#         return nums + nums

# nums = [1, 2, 1]
# solution = Solution()
# print(solution.getConcatenation(nums))

# ----------------------Количество пар
# class Solution:
#     def numIdenticalPairs(self, nums: List[int]) -> int:
#         z = 0
#         for i in range(0, len(nums)):
#             for j in range(i+1, len(nums)):
#                 if nums[i] == nums[j]:
#                     z += 1
#         return z


# nums = [1, 1, 1, 1]
# solution = Solution()
# print(solution.numIdenticalPairs(nums))


# ----------------КОличество пар соотношений прямоугольников
# from typing import List
# from collections import defaultdict


# class Solution:
# def InterchangeableRectangles(self, rectangles: List[List[int]]) -> int:
# z = 0
# for i in range(len(rectangles)):
#     for j in range(i+1, len(rectangles)):
#         if rectangles[i][0]/rectangles[i][1] == rectangles[j][0]/rectangles[j][1]:
#             z += 1
# return z

#     def InterchangeableRectangles(self, rectangles: List[List[int]]) -> int:
#         ratios = defaultdict(int)
#         count = 0
#         for width, height in rectangles:
#             ratio = width / height
#             count += ratios[ratio]
#             ratios[ratio] += 1
#         return count


# rectangles = [[4, 8], [3, 6], [10, 20], [15, 30]]
# solution = Solution()
# print(solution.InterchangeableRectangles(rectangles))


# ---------------- Подсчет неправильных пар------------------
# from typing import List
# from collections import defaultdict
    # def countBadPairs(self, nums: List[int]) -> int:
    #     count = 0
    #     diff_dict = {}
    #     for j, num in enumerate(nums):
    #         diff = j - num
    #         if diff in diff_dict:
    #             count += diff_dict[diff]
    #             diff_dict[diff] += 1
    #         else:
    #             diff_dict[diff] = 1
    #     return count
# import collections
# -----------------Второе решение---------------
# class Solution:
#     def countBadPairs(self, nums: List[int]) -> int:
#         sum = collections.defaultdict(int)
#         bad_pairs = 0
#         for i, num in enumerate(nums):
#             req_sum = i - num
#             bad_pairs += i - sum[req_sum]
#             sum[req_sum] += 1
#             print(sum.items())
#         return bad_pairs


# nums = [4,1,3,3] # 5
# # nums = [1,2,3,4,5] # 0
# solution = Solution()
# print(solution.countBadPairs(nums))


# -----------ПОДСЧЕТ КОЛИЧЕСТВА ТРОЕК-------------
# from typing import List

# class Solution:
#     def arithmeticTriplets(self, nums: List[int], diff: int) -> int:
#         count = 0
#         for i in range(len(nums)):
#             for j in range(len(nums)):
#                 for k in range(len(nums)):
#                      if (nums[i] < nums[j] < nums[k]) and (nums[j] - nums[i] == diff) and (nums[k] - nums[j] == diff):
#                         count += 1
#         return count

# nums = [0, 1, 4, 6, 7, 10]  
# diff = 3 # ожидаемый результат: 2
# solution = Solution()
# print(solution.arithmeticTriplets(nums, diff))


# ---------ОБЪЕДИНЕНИЕ ДВУХ СПИСКОВ---------------------
# from typing import Optional

# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# class Solution:
#     @staticmethod
#     def mergeTwoLists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
#         dummy = ListNode()
#         current = dummy

#         while list1 and list2:
#             if list1.val < list2.val:
#                 current.next = list1
#                 list1 = list1.next
#             else:
#                 current.next = list2
#                 list2 = list2.next
#             current = current.next

#         if list1:
#             current.next = list1
#         elif list2:
#             current.next = list2

#         return dummy.next

# # Пример использования
# list1 = ListNode(1, ListNode(2, ListNode(4)))
# list2 = ListNode(1, ListNode(3, ListNode(4)))

# # Создаем экземпляр класса Solution
# solution = Solution()

# # Вызываем метод mergeTwoLists на экземпляре класса Solution
# result = solution.mergeTwoLists(list1, list2)

# # Выводим результат
# while result:
#     print(result.val, end=", ")
#     result = result.next


# ------------ИСКЛЮЧЕНИЕ НЕНУЖНЫХ ЦИФР-----------------
# from typing import List
# class Solution:
#     def removeElement(self, nums: List[int], val: int) -> int:
#         sum = len(nums)
#         povtor = 1
#         while povtor != 0:
#             povtor = 0
#             for i in range(len(nums)):
#                 if nums[i] == val:
#                     sum -= 1
#                     povtor += 1
#                     nums.pop(i)
#                     nums.append("_")               
#         return sum
        
# # nums = [3, 2, 2, 3]
# nums = [0,1,2,2,3,0,4,2]
# val = 2
# solution = Solution()
# result = solution.removeElement(nums, val)
# print(result)
# print(nums)



# ------------ИСКЛЮЧЕНИЕ Дубликатов-----------------
# from typing import List
# class Solution:
#     def removeDuplicates(self, nums: List[int]) -> int:
#         exit = len(set(nums))
#         povtor = 1
#         while povtor != 0:
#             povtor = 0
#             for i in range(len(nums)-1):
#                 if nums[i] == nums[i+1] and nums[i] != '_':
#                     nums.pop(i)
#                     nums.append("_")
#                     povtor += 1
#                     break     
#         return exit
        
# # nums = [3, 2, 2, 3]
# nums = [0,0,0,0,2]
# # nums = [0,0,1,1,1,2,2,3,3,4]
# solution = Solution()
# result = solution.removeDuplicates(nums)
# print(result)
# print(nums)


# ---------------ПРЕОБРАЗОВАНИЕ Цельсий в Кельвин, Фаренгейт-------------
# from typing import List
# class Solution:
#     def convertTemperature(self, celsius: float) -> List[float]:
#         kelvin = celsius + 273.15
#         forengeyt = celsius * 1.80 + 32.00
#         exit = [kelvin, forengeyt]
#         return exit 
        
        
# celsius = 36.50
# solution = Solution()
# result = solution.convertTemperature(celsius)
# print(result)


# ---------ВЫВОД НОК ----------------
# class Solution:
#     def smallestEvenMultiple(self, n: int) -> int:
#         return (n%2 + 1)*n
        
# n = 5
# solution = Solution()
# print(solution.smallestEvenMultiple(n))


# -------------Поиск Наибольшего делителя------------------
from typing import List
class Solution:
    def findGCD(self, nums: List[int]) -> int:
        mi = min(nums)
        ma = max(nums)
        for i in range(ma, 0, -1):
            if ma%i == 0 and mi%i == 0:
                return i
    
        
n = [2,7,5,6,8,3,10]
solution = Solution()
print(solution.findGCD(n))