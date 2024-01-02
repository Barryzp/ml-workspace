# Example 2D array A
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Define row and column ranges
start_row, end_row = 1, 4  # Example row range
start_col, end_col = 1, 5  # Example column range

# Define the fill value for out-of-bounds indices
fill_value = -1

# Create the 2D array B
B = [[A[i][j] if i < len(A) and j < len(A[i]) else fill_value 
      for j in range(start_col, end_col)] 
     for i in range(start_row, end_row)]

for row in B:
    print(row)
