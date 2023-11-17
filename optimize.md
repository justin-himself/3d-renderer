# basics

- identify anything above O(n^2) and try to replace it to O(n)
- avoid doing repetition (sin(x) -> sx)

- Use a profiler and identify the most important contributor to cumtime
- Identify bottleneck. Common bottle neck (disk, io, memory)
- Use parallel computation

# NP Specific

- Using Numpy built in functions, instead of python loops
- deleted most of np.copy method to reduce memory io delay

- Todo: remove all np.copy and make them views instead

# 3D Specific 

- use clipping
- use z-buffering
- Can we reduce the number of polygons displayed when zoom out?
- Normal is recalculated every time, can we apply changes to normal as well?