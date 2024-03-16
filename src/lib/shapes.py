def center_of_rectangle(point1: tuple[int, int], point2: tuple[int, int]):
    """
    Returns the center of a rectangle defined by two points.
    """
    return (
        int((point1[0] + point2[0]) // 2),
        int((point1[1] + point2[1]) // 2)
    )
