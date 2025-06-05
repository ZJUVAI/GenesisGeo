def translate_constrained_to_constructive(
    point: str, predicate: str, args: list[str]
) -> tuple[str, list[str]]:
    """ Translate a predicate into construction
    
    Args:
        point: str: name of the new point
        predicate: str: name of the predicates, e.g., perp, para, etc.
        args: list[str]: list of predicate args.
    
    Return:
        (predicate, args): translated to constructive predicate.
    """
    # 直线垂直
    if predicate == 'perp':
        a, b, c, d = args
        if point in [c, d]:
            a, b, c, d = c, d, a, b
        if point == b:
            a, b = b, a
        if point == d:
            c, d = d, c
        if a == c and a == point:
            return 'on_dia', [a, b, d]
        return 'on_tline', [a, b, c, d]

    # 直线平行
    elif predicate == 'para':
        a, b, c, d = args
        if point in [c, d]:
            a, b, c, d = c, d, a, b
        if point == b:
            a, b = b, a
        return 'on_pline', [a, b, c, d]

    # 全等/等距
    elif predicate == 'cong':
        a, b, c, d = args
        if a == c and a == point:
            return 'on_bline', [a, b, d]
        if point in [c, d]:
            a, b, c, d = c, d, a, b
        if point == b:
            a, b = b, a
        if point == d:
            c, d = d, c
        if b in [c, d]:
            if b == d:
                c, d = d, c
            return 'on_circle', [a, b, d]
        return 'eqdistance', [a, b, c, d]

    # 共线
    elif predicate == 'coll':
        a, b, c = args
        if point == b:
            a, b = b, a
        if point == c:
            a, b, c = c, a, b
        return 'on_line', [a, b, c]

    # 等角
    elif predicate == 'eqangle':
        a, b, c, d, e, f = args
        if point in [d, e, f]:
            a, b, c, d, e, f = d, e, f, a, b, c
        x, b2, y, c2, d2 = b, c, e, d, f
        if point == b2:
            a, b2, c2, d2 = b2, a, d2, c2
        if point == d2 and x == y:
            return 'angle_bisector', [point, b2, x, c2]
        if point == x:
            return 'eqangle3', [x, a, b2, y, c2, d2]
        return 'on_aline', [a, x, b2, c2, y, d2]

    # 四点共圆
    elif predicate == 'cyclic':
        a, b, c = [x for x in args if x != point]
        return 'on_circum', [point, a, b, c]

    # 其它直接返回
    return predicate, [point] + args if point not in args else args