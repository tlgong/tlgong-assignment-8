import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.spatial.distance import cdist
import os


result_dir = "results"
os.makedirs(result_dir, exist_ok=True)


def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8],
                                  [cluster_std * 0.8, cluster_std]])

    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    X2 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    direction = np.array([1, 1])

    unit_direction = direction / np.linalg.norm(direction)

    x = distance


    X2 = X2 + x * unit_direction
    y2 = np.ones(n_samples)


    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y


# 拟合逻辑回归并提取系数
def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2


def do_experiments(start, end, step_num):

    shift_distances = np.linspace(start, end, step_num)
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list, margin_widths = [], [], [], [], [], [], []
    sample_data = {}

    n_samples = 100
    n_cols = 4
    n_rows = (step_num + n_cols - 1) // n_cols
    plt.figure(figsize=(n_cols * 5, n_rows * 5))

    for i, distance in enumerate(shift_distances, 1):
        X, y = generate_ellipsoid_clusters(distance=distance, n_samples=n_samples)
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)

        loss = log_loss(y, model.predict_proba(X)[:, 1])
        loss_list.append(loss)


        if beta2 != 0:
            slope = -beta1 / beta2
            intercept = -beta0 / beta2
        else:
            slope = np.inf
            intercept = np.inf
        slope_list.append(slope)
        intercept_list.append(intercept)


        plt.subplot(n_rows, n_cols, i)
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0', alpha=0.5)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1', alpha=0.5)


        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0.5], colors='green')

        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]
        for level, alpha in zip(contour_levels, alphas):
            class_1_contour = plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            class_0_contour = plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)


        try:
            paths_class1 = class_1_contour.collections[0].get_paths()
            paths_class0 = class_0_contour.collections[0].get_paths()
            if paths_class1 and paths_class0:
                vertices_class1 = paths_class1[0].vertices
                vertices_class0 = paths_class0[0].vertices
                distances = cdist(vertices_class1, vertices_class0, metric='euclidean')
                min_distance = np.min(distances)
                margin_widths.append(min_distance)
            else:
                min_distance = np.nan
                margin_widths.append(min_distance)
        except IndexError:
            min_distance = np.nan
            margin_widths.append(min_distance)

        plt.title(f"Shift Distance = {distance:.2f}", fontsize=12)
        plt.xlabel("x1")
        plt.ylabel("x2")

        equation_text = f"{beta0:.2f} + {beta1:.2f} * x1 + {beta2:.2f} * x2 = 0\nx2 = {slope:.2f} * x1 + {intercept:.2f}"
        margin_text = f"Margin Width: {min_distance:.2f}" if not np.isnan(min_distance) else "Margin Width: N/A"
        plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=8,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))
        plt.text(0.05, 0.85, margin_text, transform=plt.gca().transAxes, fontsize=8,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

        if i == 1:
            plt.legend(loc='upper left', fontsize=8)

        sample_data[distance] = (X, y, model, beta0, beta1, beta2, min_distance)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")
    plt.close()


    plt.figure(figsize=(18, 15))

    # Plot Beta0
    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list, marker='o')
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    # Plot Beta1
    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list, marker='o', color='orange')
    plt.title("Shift Distance vs Beta1 (Coefficient for x1)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    # Plot Beta2
    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list, marker='o', color='green')
    plt.title("Shift Distance vs Beta2 (Coefficient for x2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    # Plot Slope (Beta1 / Beta2)
    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list, marker='o', color='red')
    plt.title("Shift Distance vs Slope (-Beta1/Beta2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope")
    plt.ylim(-5, 5)  # 根据需要调整y轴范围

    # Plot Intercept Ratio (Beta0 / Beta2)
    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_list, marker='o', color='purple')
    plt.title("Shift Distance vs Intercept (-Beta0/Beta2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Intercept")

    # Plot Logistic Loss
    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list, marker='o', color='brown')
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    # Plot Margin Width
    plt.subplot(3, 3, 7)
    plt.plot(shift_distances, margin_widths, marker='o', color='cyan')
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")
    plt.close()

    print(f"graph stores in '{result_dir}' 中。")


if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
