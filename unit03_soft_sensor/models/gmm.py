import numpy as np
from sklearn.mixture import GaussianMixture


class GMMMeanRegressor:
    """GMM-based regression baseline。

    GMM 本身是无监督密度模型，不直接输出连续 y。
    这里的回归逻辑是：
    1. 用 X_train 拟合 GMM，把样本划分到若干隐含工况/组件。
    2. 每个组件只用有标签训练样本计算 y 均值。
    3. 预测时使用 posterior_probability @ component_y_mean。
    """

    def __init__(self, n_components, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
            reg_covar=1e-6,
            n_init=3,
        )
        self.component_y_mean_ = None

    def fit(self, X_train, y_train, labeled_mask):
        self.gmm.fit(X_train)

        X_labeled = X_train[labeled_mask]
        y_labeled = y_train[labeled_mask].reshape(-1)
        global_mean = float(np.mean(y_labeled))

        labeled_components = self.gmm.predict(X_labeled)
        component_means = []

        for component_id in range(self.n_components):
            component_mask = labeled_components == component_id
            if np.any(component_mask):
                component_means.append(float(np.mean(y_labeled[component_mask])))
            else:
                # 若某个 GMM 组件没有有标签样本，用全局有标签均值兜底。
                component_means.append(global_mean)

        self.component_y_mean_ = np.asarray(component_means, dtype=np.float32).reshape(-1, 1)
        return self

    def predict(self, X):
        posterior = self.gmm.predict_proba(X)
        return posterior @ self.component_y_mean_

