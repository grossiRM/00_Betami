def well_fxn(u,tol):
    """
    Returns the value of the well function in the Theis equation to specified accuracy.

    Parameter
    ---------
    u: float
        The value of u in the Theis equation, u = r^2*S/4*pi*T
    tol: float
        Accuracy of calculation
    """
    
    import numpy as np
    import math

    # Initalize well function
    wu = -0.57721566 - np.log(u)
    i = 1
    err =  tol + 1.0
    
    # Iteratively calculate value of infinite series     
    while err>tol:
        if (i % 2) == 0:
            wu_new = wu - u**i/(i*math.factorial(i))
        else:
            wu_new = wu + u**i/(i*math.factorial(i))
        err = abs(wu_new - wu)
        wu = wu_new
        i = i + 1
     
    return wu
