import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def get_conf_intercept(alpha, lr, X, y):
    """
    Returns (1-alpha) 2-sided confidence intervals
    for sklearn.LinearRegression coefficients
    as a pandas DataFrame
    """
    coefs = np.r_[[lr.intercept_], lr.coef_]
    X_aux = np.zeros((X.shape[0], X.shape[1] + 1))
    X_aux[:, 1:] = X
    X_aux[:, 0] = 1
    dof = -np.diff(X_aux.shape)[0]
    mse = np.sum((y - lr.predict(X)) ** 2) / dof
    var_params = np.diag(np.linalg.inv(X_aux.T.dot(X_aux)))
    t_val = stats.t.isf(alpha / 2, dof)
    gap = t_val * np.sqrt(mse * var_params)
    return {
        "coeffs": coefs,
        "lower": coefs - gap,
        "upper": coefs + gap
    }


def plotComparisonIntercept(X, Y, ax):
    lin_model = LinearRegression().fit(X, Y)
    line_x = np.linspace(0, np.max(X)).reshape(-1, 1)
    line_y = lin_model.predict(line_x)
    # Confident intervals
    cf = get_conf_intercept(0.05, lin_model, X, Y)
    slope_interval = [cf["lower"][1], cf["upper"][1]]
    intercept_interval = [cf["lower"][0], cf["upper"][0]]
    r2_score = lin_model.score(X, Y)
    ax.plot(line_x, line_y, 'r',
                     label='y={:.2f}x+{:.2f} \n slope: [{:.2f}, {:.2f}] \n intercept: [{:.2f}, {:.2f}]  '
                           '\n $R^2$ = {:.2f}'.format(cf["coeffs"][1], cf["coeffs"][0], slope_interval[0], slope_interval[1],
                                                      intercept_interval[0], intercept_interval[1], r2_score))
    ax.legend(loc='lower right')


def plotPolygons(polygon_list, ax, color):
    for current_polygon in polygon_list:
        if current_polygon.geom_type == "MultiPolygon":
            for geom in current_polygon.geoms:
                xs, ys = geom.exterior.xy
                ax.plot(xs, ys, color)
        else:
            xs, ys = current_polygon.exterior.xy
            ax.plot(xs, ys, color)