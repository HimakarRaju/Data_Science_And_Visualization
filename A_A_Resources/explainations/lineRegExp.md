# linear Regression

Imagine you have two groups of numbers:

- **x = [1, 2, 3]**
- **y = [2, 4, 5]**

We want to find a straight line that best fits these points.

---

## Step 1: Find the mean of x and y

- **Mean of x**: Add the x numbers and divide by how many there are.
  - Mean of x = (1 + 2 + 3) / 3 = 6 / 3 = **2**

- **Mean of y**: Add the y numbers and divide by how many there are.
  - Mean of y = (2 + 4 + 5) / 3 = 11 / 3 = **3.67**

---

### Step 2: Find the slope (m) using the formula

We need to calculate two things:

1. **sum_xy**: This is where we multiply the differences of each x and y from their means, and then add them up.

    - For x = 1, y = 2: \((1 - 2) *(2 - 3.67) = (-1)* (-1.67) = 1.67\)
    - For x = 2, y = 4: \((2 - 2) *(4 - 3.67) = (0)* (0.33) = 0\)
    - For x = 3, y = 5: \((3 - 2) *(5 - 3.67) = (1)* (1.33) = 1.33\)

    Now, add them up:
    \[sum_xy = 1.67 + 0 + 1.33 = **3**\]

2. **sum_xx**: This is where we square the differences of each x from the mean of x, and add them up.

    - For x = 1: \((1 - 2)² = (-1)² = 1\)
    - For x = 2: \((2 - 2)² = (0)² = 0\)
    - For x = 3: \((3 - 2)² = (1)² = 1\)

    Now, add them up:
    \[
    sum_xx = 1 + 0 + 1 = **2**
    \]

Now, calculate the **slope (m)**:
\[
m = \frac{sum\_xy}{sum\_xx} = \frac{3}{2} = **1.5**
\]

---

### Step 3: Find the intercept (b)

The intercept is calculated using this formula:
\[
b = mean\_y - (m * mean\_x)
\]

Now, plug in the numbers:
\[
b = 3.67 - (1.5 * 2) = 3.67 - 3 = **0.67**
\]

---

### Step 4: Write the line equation

Now that we have the slope (m = 1.5) and intercept (b = 0.67), the equation of the line is:
\[y = 1.5 * x + 0.67\]

---

### Step 5: Use this equation to predict y for any x

For example, if you want to know the y value for x = 4, just plug it into the equation:
\[y = 1.5 * 4 + 0.67 = 6 + 0.67 = **6.67**\]

---

### Recap

- **Slope (m)** = 1.5 (The line goes up by 1.5 for each increase of 1 in x).
- **Intercept (b)** = 0.67 (This is where the line crosses the y-axis when x is 0).
- The line equation is: **y = 1.5 * x + 0.67**.

---
