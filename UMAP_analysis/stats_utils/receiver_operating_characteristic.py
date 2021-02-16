
def true_positive_rate(truth, predicted):
    return sum(truth & predicted) / sum(truth)

def false_positive_rate(truth, predicted):
    return sum(~truth & predicted) / sum(~truth)

def roc_curve(truth, predicted_values, steps = 500):

    x = []
    y = []

    max_p = max(predicted_values)
    min_p = min(predicted_values)

    predicted = predicted_values >= min_p
    x.append(false_positive_rate(truth, predicted))
    y.append(true_positive_rate(truth, predicted))

    for threshold in np.linspace(min_p, max_p, steps):
        predicted = predicted_values > threshold
        x.append(false_positive_rate(truth, predicted))
        y.append(true_positive_rate(truth, predicted))

    df = pd.DataFrame([x, y]).T
    df.columns = ["FPR", "TPR"]
    df["threshold"] = [min_p]+list(np.linspace(min_p, max_p, steps))
    return df


def make_roc_fig(moa):
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(df.propagation_FPR, df.propagation_TPR, c = 'k')
    ax.plot(df.random_forest_FPR, df.random_forest_TPR, c = '#00c2d7')
    ax.plot(df.nearest_neighbor_FPR, df.nearest_neighbor_TPR, c =  '#ae342b')
    ax.set_title(" ".join([x.title() for x in moa.split("_")]))
    ax.set_xlabel("False Positive Rate", fontsize = 10)
    ax.set_ylabel("True Positive Rate",  fontsize = 10)
    return fig


def auc_from_roc(x_curve,y_curve):

    auc = 0
    for i,x,y in zip(range(len(x_curve)), x_curve, y_curve):

        if i == 0:
            x_prev = x
            y_prev = y
            continue

        auc += (x_prev - x) * (y+y_prev)/2
        x_prev = x
        y_prev = y

    return auc
