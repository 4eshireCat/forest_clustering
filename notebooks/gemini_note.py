import json
import os

# Базовый шаблон метаданных для Jupyter Notebook
NB_METADATA = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def create_notebook(cells, filename):
    notebook = NB_METADATA.copy()
    notebook["cells"] = cells
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f" Сгенерирован файл: {filename}")

# --- НОУТБУК 1: БАЗОВЫЙ ПАЙПЛАЙН ---
cells_nb1 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Базовое использование библиотеки ForestClustering\n",
            "В этом ноутбуке показан стандартный процесс кластеризации данных с помощью случайного леса без явного использования объектов `KMeans`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from sklearn.datasets import make_blobs\n",
            "from sklearn.metrics import silhouette_score\n",
            "\n",
            "# Импорт целевой библиотеки\n",
            "from forest_clustering import ForestClustering"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Генерация синтетических данных"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)\n",
            "\n",
            "plt.figure(figsize=(8, 5))\n",
            "plt.scatter(X[:, 0], X[:, 1], s=20, color='gray', alpha=0.7)\n",
            "plt.title(\"Исходные неразмеченные данные\")\n",
            "plt.grid(True)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Инициализация и обучение модели\n",
            "Используем строковый идентификатор `'kmeans'` для передачи в параметр `clusterer`, избегая прямой передачи класса."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "model = ForestClustering(\n",
            "    n_estimators=100,\n",
            "    max_depth=8,\n",
            "    clusterer='kmeans',  # Передача в виде строки\n",
            "    n_clusters=4,\n",
            "    random_state=42\n",
            ")\n",
            "\n",
            "# Обучение и получение меток\n",
            "labels = model.fit_predict(X)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Визуализация и валидация"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(8, 5))\n",
            "plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20, alpha=0.8)\n",
            "plt.title(\"Результат кластеризации (ForestClustering)\")\n",
            "plt.grid(True)\n",
            "plt.show()\n",
            "\n",
            "score = silhouette_score(X, labels)\n",
            "print(f\"Silhouette Score для полученной разметки: {score:.4f}\")"
        ]
    }
]

# --- НОУТБУК 2: ПРОДВИНУТЫЙ ФУНКЦИОНАЛ ---
cells_nb2 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Продвинутые возможности ForestClustering\n",
            "Изучаем работу с реальными признаками, извлечение матрицы близости (Proximity Matrix) и Unsupervised Feature Importance. В качестве кластеризатора используем иерархический метод."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import seaborn as sns\n",
            "import matplotlib.pyplot as plt\n",
            "from sklearn.datasets import load_wine\n",
            "from sklearn.cluster import AgglomerativeClustering\n",
            "\n",
            "from forest_clustering import ForestClustering"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Загрузка данных"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "wine = load_wine()\n",
            "X = wine.data\n",
            "feature_names = wine.feature_names\n",
            "print(f\"Размерность матрицы признаков: {X.shape}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Настройка модели со сторонним алгоритмом (без KMeans)\n",
            "Передаем объект `AgglomerativeClustering` и включаем расчет матрицы близости объектов."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "custom_clusterer = AgglomerativeClustering(n_clusters=3, linkage='ward')\n",
            "\n",
            "advanced_model = ForestClustering(\n",
            "    n_estimators=250,\n",
            "    min_samples_leaf=4,\n",
            "    clusterer=custom_clusterer, \n",
            "    compute_proximity=True,      # Запрос на расчет матрицы близости\n",
            "    n_jobs=-1,\n",
            "    random_state=42\n",
            ")\n",
            "\n",
            "labels_advanced = advanced_model.fit_predict(X)\n",
            "print(\"Модель успешно обучена.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Визуализация Proximity Matrix\n",
            "Показывает, насколько часто пары объектов оказывались в одних листьях деревьев."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if hasattr(advanced_model, 'proximity_matrix_'):\n",
            "    prox_matrix = advanced_model.proximity_matrix_\n",
            "    \n",
            "    plt.figure(figsize=(10, 8))\n",
            "    # Отобразим срез матрицы для первых 60 объектов\n",
            "    sns.heatmap(prox_matrix[:60, :60], cmap='rocket_r', square=True)\n",
            "    plt.title(\"Матрица близости (Фрагмент 60x60)\")\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"Атрибут proximity_matrix_ отсутствует. Проверьте настройки API.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Оценка важности признаков\n",
            "Выделение наиболее информативных параметров в режиме обучения без учителя."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if hasattr(advanced_model, 'feature_importances_'):\n",
            "    importances = advanced_model.feature_importances_\n",
            "    indices = np.argsort(importances)[::-1]\n",
            "    \n",
            "    plt.figure(figsize=(12, 6))\n",
            "    plt.title(\"Важность признаков (Unsupervised Feature Importance)\")\n",
            "    plt.bar(range(X.shape[1]), importances[indices], align=\"center\", color='darkslateblue')\n",
            "    plt.xticks(range(X.shape[1]), np.array(feature_names)[indices], rotation=45, ha='right')\n",
            "    plt.grid(axis='y', linestyle='--')\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"Атрибут feature_importances_ не поддерживается данной конфигурацией модели.\")"
        ]
    }
]

if __name__ == "__main__":
    create_notebook(cells_nb1, "01_forest_clustering_basic.ipynb")
    create_notebook(cells_nb2, "02_forest_clustering_advanced.ipynb")
    print("\n Готово! Открывай Jupyter Lab или VS Code и запускай получившиеся файлы.")