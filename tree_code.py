import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Находит оптимальный порог для разбиения вектора признака по критерию Джини.

    Критерий Джини определяется следующим образом:
    .. math::
        Q(R) = -\\frac {|R_l|}{|R|}H(R_l) -\\frac {|R_r|}{|R|}H(R_r),

    где:
    * :math:R — множество всех объектов,
    * :math:R_l и :math:R_r — объекты, попавшие в левое и правое поддерево соответственно.

    Функция энтропии :math:H(R):
    .. math::
        H(R) = 1 - p_1^2 - p_0^2,

    где:
    * :math:p_1 и :math:p_0 — доля объектов класса 1 и 0 соответственно.

    Указания:
    - Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    - В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака.
    - Поведение функции в случае константного признака может быть любым.
    - При одинаковых приростах Джини нужно выбирать минимальный сплит.
    - Для оптимизации рекомендуется использовать векторизацию вместо циклов.

    Parameters
    ----------
    feature_vector : np.ndarray
        Вектор вещественнозначных значений признака.
    target_vector : np.ndarray
        Вектор классов объектов (0 или 1), длина feature_vector равна длине target_vector.

    Returns
    -------
    thresholds : np.ndarray
        Отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на
        два различных поддерева.
    ginis : np.ndarray
        Вектор со значениями критерия Джини для каждого порога в thresholds.
    threshold_best : float
        Оптимальный порог для разбиения.
    gini_best : float
        Оптимальное значение критерия Джини.

    """
    # Проверяем, что все значения признака не одинаковы
    if len(np.unique(feature_vector)) <= 1:
        return np.array([]), np.array([]), 0, 0

    # Сортируем индексы по значениям признака
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    # Находим уникальные значения и их позиции
    unique_values = np.unique(sorted_features)

    # Создаем пороги как среднее между соседними уникальными значениями
    thresholds = []
    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i + 1]) / 2
        thresholds.append(threshold)

    thresholds = np.array(thresholds)

    if len(thresholds) == 0:
        return np.array([]), np.array([]), 0, 0

    # Вычисляем критерий Джини для каждого порога
    ginis = []
    n_total = len(target_vector)

    for threshold in thresholds:
        # Разделяем на левое и правое поддерево
        left_mask = feature_vector < threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        # Пропускаем пороги, которые приводят к пустым поддеревьям
        if n_left == 0 or n_right == 0:
            ginis.append(-np.inf)
            continue

        # Вычисляем энтропию Джини для левого поддерева
        left_targets = target_vector[left_mask]
        p1_left = np.mean(left_targets)
        p0_left = 1 - p1_left
        h_left = 1 - p1_left**2 - p0_left**2

        # Вычисляем энтропию Джини для правого поддерева
        right_targets = target_vector[right_mask]
        p1_right = np.mean(right_targets)
        p0_right = 1 - p1_right
        h_right = 1 - p1_right**2 - p0_right**2

        # Вычисляем критерий Джини (с минусом, так как мы максимизируем прирост информации)
        gini = -(n_left / n_total) * h_left - (n_right / n_total) * h_right
        ginis.append(gini)

    ginis = np.array(ginis)

    # Находим лучший порог (максимальный критерий Джини)
    valid_indices = ginis != -np.inf
    if not np.any(valid_indices):
        return thresholds, ginis, thresholds[0], ginis[0]

    valid_ginis = ginis[valid_indices]
    valid_thresholds = thresholds[valid_indices]

    best_idx = np.argmax(valid_ginis)
    threshold_best = valid_thresholds[best_idx]
    gini_best = valid_ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
    ):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        """
        Обучение узла дерева решений.

        Если все элементы в подвыборке принадлежат одному классу, узел становится терминальным.

        Parameters
        ----------
        sub_X : np.ndarray
            Подвыборка признаков.
        sub_y : np.ndarray
            Подвыборка меток классов.
        node : dict
            Узел дерева, который будет заполнен информацией о разбиении.
        depth : int
            Текущая глубина узла.

        """
        # Проверяем условия остановки
        if len(sub_y) == 0:
            node["type"] = "terminal"
            node["class"] = 0  # Значение по умолчанию для пустого множества
            return

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        # Проверяем ограничения по глубине и количеству объектов
        if (self._max_depth is not None and depth >= self._max_depth) or (
            self._min_samples_split is not None and len(sub_y) < self._min_samples_split
        ):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {key: clicks.get(key, 0) / count for key, count in counts.items()}
                sorted_categories = sorted(ratio, key=ratio.get)
                categories_map = {category: i for i, category in enumerate(sorted_categories)}
                feature_vector = np.vectorize(categories_map.get)(sub_X[:, feature])
            else:
                raise ValueError("Некорректный тип признака")

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                threshold_best = threshold

                if feature_type == "real":
                    split = sub_X[:, feature] < threshold
                elif feature_type == "categorical":
                    # Исправлено: используем правильный порог для категориальных признаков
                    split_point = int(np.round(threshold))
                    if split_point <= 0:
                        split_point = 1
                    elif split_point >= len(sorted_categories):
                        split_point = len(sorted_categories) - 1

                    left_categories = sorted_categories[:split_point]
                    split = np.isin(sub_X[:, feature], left_categories)

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # Проверяем, что разбиение не создает пустые поддеревья
        if np.sum(split) == 0 or np.sum(~split) == 0:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # Проверяем ограничения min_samples_leaf
        if self._min_samples_leaf is not None:
            if np.sum(split) < self._min_samples_leaf or np.sum(~split) < self._min_samples_leaf:
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]
                return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            # Исправлено: сохраняем правильные категории для разбиения
            feature_type = self._feature_types[feature_best]
            if feature_type == "categorical":
                counts = Counter(sub_X[:, feature_best])
                clicks = Counter(sub_X[sub_y == 1, feature_best])
                ratio = {key: clicks.get(key, 0) / count for key, count in counts.items()}
                sorted_categories = sorted(ratio, key=ratio.get)

                split_point = int(np.round(threshold_best))
                if split_point <= 0:
                    split_point = 1
                elif split_point >= len(sorted_categories):
                    split_point = len(sorted_categories) - 1

                node["categories_split"] = sorted_categories[:split_point]
        else:
            raise ValueError("Некорректный тип признака")

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        """
        Рекурсивное предсказание класса для одного объекта по узлу дерева решений.

        Если узел терминальный, возвращается предсказанный класс.
        Если узел не терминальный, выборка передается в соответствующее поддерево для дальнейшего предсказания.

        Parameters
        ----------
        x : np.ndarray
            Вектор признаков одного объекта.
        node : dict
            Узел дерева решений.

        Returns
        -------
        int
            Предсказанный класс объекта.
        """
        # Если узел терминальный, возвращаем его класс
        if node["type"] == "terminal":
            return node["class"]

        # Если узел не терминальный, определяем, в какое поддерево направить объект
        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]

        if feature_type == "real":
            # Для вещественных признаков сравниваем с порогом
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        elif feature_type == "categorical":
            # Для категориальных признаков проверяем принадлежность к множеству
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Некорректный тип признака")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
