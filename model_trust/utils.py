from itertools import chain

def reorder_metrics (df, metric):
    metric_new = []
    for sub_metric in metric:
        if len(sub_metric) < len(df['Topic'].unique()):
            sub_metric.extend([0.2]*(len(df['Topic'].unique())-len(sub_metric)))
            metric_new.append(list(chain(sub_metric)))
        else:
            metric_new.append(list(chain(sub_metric)))
    return metric_new