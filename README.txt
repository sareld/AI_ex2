200878189
203516083
***
Comments:

Evaluation function description:
    In every state we check three criteria:
    1. empty - we try to maximize the number of empty tiles
    2. smoothness - we try to minimize the the sum of differences between side-by-side tiles.
        For example, the smoothness of the board below
        [2, 4, 0, 0]
        [4, 4, 16,0]
        [4, 2, 32,0]
        [2, 2, 16,0]
        would be :
        2+4(first row) + 0+12+16 (second row) + 2+30+32 (third row) +0+14+16 (forth row) + 2+0+2 (first column )+ ...
        in order to minimize this number by max player we take the minus of it.
        we use convolution on rows and columns with kernel [1,-1] to calculate the diffs. (first differential).
     3. monotonicity - we try to maximize the monotonicity, it means we count how many sorting violations there are in
     every row and column, in other words how many times we change from going up to going down and vice-versa.
     in order to do that we use the same convolution on the signs of the first convolution and we sum all.
     we want to maximize the monotonicity by minimizing the violations, so we take the minus of that value.

     the total evaluation is a the weighted sum of all three criteria we found a good solution by taking:
     10*(empty) + 1*(smoothness) +1*(monotonicity)
