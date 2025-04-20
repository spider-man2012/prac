# A naive recursive Python program to find the minimum number
# of operations to convert str1 to str2

def editDistance(str1, str2, m, n):
    # If first string is empty, insert all characters of second string
    if m == 0:
        return n
    
    # If second string is empty, remove all characters of first string
    if n == 0:
        return m
    
    # If last characters of both strings are the same, ignore them and recurse for the remaining part
    if str1[m - 1] == str2[n - 1]:
        return editDistance(str1, str2, m - 1, n - 1)
    
    # If last characters are not the same, consider all three operations:
    # Insert, Remove, and Replace. Compute the minimum cost among them.
    return 1 + min(
        editDistance(str1, str2, m, n - 1),    # Insert
        editDistance(str1, str2, m - 1, n),    # Remove
        editDistance(str1, str2, m - 1, n - 1) # Replace
    )

# Driver code
str1 = "sunday"
str2 = "saturday"
print('Edit Distance is:', editDistance(str1, str2, len(str1), len(str2)))
