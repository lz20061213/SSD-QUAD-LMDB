# --------------------------------------------------------
# SSD for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# Modified by Zhuang Liu
# --------------------------------------------------------

def GetProperty(kwargs, name, default):
    return kwargs[name] \
        if name in kwargs else default

def pointCmp(pointA, pointB, center):
    if pointA[0] >= 0 and pointB[0] <= 0:
        return True
    if pointA[0] == 0 and pointB[0] == 0:
        return pointA[1] > pointB[1]
    # cross product of OA and OB, O presents the center
    # det = ax*by-ay*bx = |a|*|b|*sin(theta)
    det = (pointA[0] - center[0]) * (pointB[1] - center[1]) - (pointA[1] - center[1]) * (pointB[0] - center[0])
    if det < 0:
        return True
    if det > 0:
        return False
    # process if det=0, we use the distance of OA, OB
    dOA = (pointA[0] - center[0]) * (pointA[0] - center[0]) + (pointA[1] - center[1]) * (pointA[1] - center[1])
    dOB = (pointB[0] - center[0]) * (pointB[0] - center[0]) + (pointB[1] - center[1]) * (pointB[1] - center[1])
    return dOA > dOB

def ReorderPoints(points):
    # here we use clockwise to sort the points
    center = [sum([p[0] for p in points]) / len(points), sum([p[1] for p in points]) / len(points)]
    # print(center)
    for i in range(len(points)):
        for j in range(len(points) - i - 1):
            # print((i, j + i + 1))
            if pointCmp(points[i], points[j + i + 1], center):
                temp = points[i]
                points[i] = points[j + i + 1]
                points[j + i + 1] = temp
    # the xmin point will be the first point
    index = 0
    fpoint = points[index]
    for i in range(1, len(points)):
        if points[i][0] < fpoint[0]:
            index = i
            fpoint = points[index]
        elif points[i][0] == fpoint[0] and points[i][1] < fpoint[1]:
            index = i
            fpoint = points[index]
    points = points[index:] + points[:index]
    return points
