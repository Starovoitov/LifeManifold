# Пакет `src.worldspace`: подробное описание

Этот документ описывает, что делает пакет **`worldspace`**, как связаны его части и что именно означают параметры вроде **`noise`** в текущей реализации. Описание привязано к коду в `src/worldspace/`.

---

## 1. Роль пакета

**Цель:** трактовать «мир» как точку в пространстве правил → прогнать простую симуляцию → получить числовой «отпечаток поведения» → спроецировать миры на плоскость и сгруппировать их.

На высоком уровне это конвейер:

```mermaid
flowchart LR
  subgraph inputs["Вход"]
    G["Генератор миров"]
  end
  subgraph core["Ядро"]
    WS["WorldSpec JSON"]
    SIM["Симулятор CA"]
    MET["Метрики M(world)"]
    EMB["PCA → 2D"]
    CLU["k-means"]
  end
  subgraph out["Выход"]
    FILE["JSONL (файл или stdout)"]
  end
  G --> WS
  WS --> SIM
  SIM --> MET
  MET --> EMB
  MET --> CLU
  EMB --> FILE
  CLU --> FILE
```

- **`WorldSpec`** — статическое описание одного мира (правила и числовые параметры).
- **`run_world`** — динамика поля и **сразу** шестимерные метрики (без длинных списков по шагам).
- **`stream_world_space_to_jsonl`** — двухпроходный прогон: PCA/k-means по memmap, запись JSONL **потоково**, память по числу миров **O(1)**.

Пакет **не зависит** от legacy-слоёв приложения (`celery`, `redis`, веб-сокеты и т.д.) и задуман как автономный исследовательский пайплайн.

---

## 2. Структура модулей

```mermaid
flowchart TB
  subgraph pkg["src/worldspace"]
    spec["spec.py — WorldSpec"]
    gen["generators.py — генераторы траекторий в пространстве миров"]
    wmath["math.py — соседи, PCA, k-means, формулы метрик"]
    sim["simulator.py — run_world"]
    met["metrics.py — WorldMetrics"]
    pipe["pipeline.py — stream_world_space_to_jsonl"]
    cli["cli.py — точка входа CLI"]
    main["__main__.py — python -m src.worldspace"]
  end
  cli --> main
  cli --> pipe
  pipe --> gen
  pipe --> sim
  pipe --> met
  sim --> wmath
  met --> wmath
  pipe --> wmath
  sim --> spec
```

| Модуль | Назначение |
|--------|------------|
| `spec.py` | Датакласс мира + сериализация JSON |
| `generators.py` | Случайные миры, random walk, цепи Маркова, заглушки NN/LLM |
| `math.py` | `neighbor_count`, PCA по достаточным статистикам (`pca_mean_and_basis_2d`, `project_pca_2d`), `binary_entropy`, `oscillation`, `pattern_diversity_from_frame` (импорт: `from . import math as ws_math`) |
| `simulator.py` | CA по шагам; онлайн-метрики и один финальный снимок сетки |
| `metrics.py` | Датакласс метрик и сериализация вектора в JSON |
| `pipeline.py` | Потоковый пайплайн: memmap метрик, PCA по достаточным статистикам, k-means по строкам, JSONL |
| `cli.py` / `__main__.py` | Запуск из командной строки |

Краткая архитектурная заметка также есть в `src/worldspace/ARCHITECTURE.md`; этот файл (**`docs/WORLDSPACE.md`**) глубже раскрывает семантику параметров и метрик.

---

## 3. Схема данных: `WorldSpec`

Один мир — это объект **`WorldSpec`**, который можно представить как JSON.

### 3.1 Поля и смысл в коде

```mermaid
classDiagram
  class WorldSpec {
    list birth
    list survival
    float noise
    float resource_regen
    float predation
    list cell_types
    str neighborhood
    int grid_size
    int steps
    int seed
  }
```

| Поле | Тип | Роль в симуляторе |
|------|-----|-------------------|
| `birth` | `list[int]` | Число живых соседей (Moore), при котором **мертвая** клетка становится живой |
| `survival` | `list[int]` | Число живых соседей, при котором **живая** клетка остаётся живой |
| `noise` | `float` | Вероятность **случайного переворота** состояния клетки после правила рождения/выживания (см. §4.3) |
| `resource_regen` | `float` | Вероятность появления «еды» на клетке за шаг + стартовая плотность еды |
| `predation` | `float` | Интенсивность вероятностной гибели при высокой плотности соседей |
| `cell_types` | `list[str]` | **Декларативный список** типов для спеки; текущий симулятор использует только бинарное `life` и слой `food` |
| `neighborhood` | `str` | В спеки по умолчанию `"moore"`; реализовано только **Moore** с тором |
| `grid_size` | `int` | Размер поля \(N \times N\) |
| `steps` | `int` | Число итераций времени |
| `seed` | `int` | Сид для `numpy.random.Generator` |

Пример JSON-образца:

```json
{
  "birth": [3],
  "survival": [2, 3],
  "noise": 0.01,
  "resource_regen": 0.02,
  "predation": 0.3,
  "cell_types": ["empty", "life", "food"],
  "neighborhood": "moore",
  "grid_size": 50,
  "steps": 300,
  "seed": 0
}
```

---

## 4. Симулятор: что происходит на каждом шаге

Функция **`run_world(world)`** поддерживает два скрытых поля состояния на сетке:

- **`life`** — \(0\) или \(1\) (мертва / жива).
- **`food`** — \(0\) или \(1\) (нет еды / есть еда).
- **`ages`** — возраст живой клетки в шагах (для метрики средней продолжительности «жизни» при гибели).

### 4.1 Инициализация

```mermaid
flowchart TD
  A["RNG := seed(world.seed)"]
  B["life ~ Bernoulli(0.2) на каждой клетке"]
  C["food ~ Bernoulli(resource_regen) на каждой клетке"]
  D["ages := 0"]
  A --> B --> C --> D
```

То есть **20%** клеток случайно живы в начале (константа в коде, не из `WorldSpec`).

### 4.2 Детерминированное ядро (birth / survival)

Для каждой клетки считается **`neighbors`** — число живых соседей по **Moore** (8 клеток), границы **торические** (`np.roll`).

$$
\text{born}(x,y) = \mathbf{1}[\text{life}=0 \land \text{neighbors} \in \text{birth}]
$$

$$
\text{survive}(x,y) = \mathbf{1}[\text{life}=1 \land \text{neighbors} \in \text{survival}]
$$

$$
\text{next\_life} = \max(\text{born}, \text{survive}) \quad \text{(побитово по клеткам)}
$$

Это обобщение «игры Жизни»: множества `birth` и `survival` задают правило целиком.

### 4.3 Что такое `noise` (шум)

После вычисления `next_life` по правилам, для **каждой клетки** независимо:

- с вероятностью **`noise`** состояние **инвертируется**: \(0 \leftrightarrow 1\).

Формально: если `flip[x,y] ~ Bernoulli(noise)`, то  
`next_life[x,y] := 1 - next_life[x,y]` при `flip`.

**Интерпретация:** это не «ошибка измерения», а **стохастический CA**: случайные мутации/радиация/микрофлуктуации правил на уровне клетки. Чем выше `noise`, тем сильнее система отталкивается от чистого правила Conway-подобной динамики.

Ограничение в генераторах может быть другим; в симуляторе значение просто должно быть адекватным для вероятности (обычно \([0, 1]\)).

### 4.4 Что такое `predation` (хищничество / давление соседей)

Если `predation > 0`:

- `exposure = neighbors / 8.0` — доля занятых соседских клеток.
- для живых клеток после шума: с вероятностью  
  **`predation * exposure`** клетка становится мёртвой.

То есть чем плотнее окружение живыми соседями, тем выше шанс «гибели от давления». Это грубая модель конкуренции/хищничества без отдельного типа «хищник».

### 4.5 Что такое `resource_regen` (ресурсы / еда)

Два использования:

1. **Инициализация:** стартовая карта еды — Bernoulli(`resource_regen`) по клеткам.
2. **Каждый шаг:** для каждой клетки независимо с вероятностью **`resource_regen`** выставляется `food = 1` (еда может «вырасти» поверх уже существующей логики).

Затем:

- если **`food == 1` и клетка живая** после всех обновлений `next_life`, еда потребляется (`food := 0`), и **`ages`** на этом шаге получает **+1 бонус** к приросту возраста (`feed_bonus`).

**Интерпретация:** еда повышает «выживаемость» возраста в смысле метрик по гибели (косвенно); отдельный тип «ресурс» на поле для правил рождения **не участвует** — только через возраст и косвенно через динамику.

### 4.6 Гибель и возраст

- Если клетка была жива (`life == 1`) и стала мёртвой (`next_life == 0`), возрасты гибелей **суммируются** в счётчиках (без списка всех `death_ages`).
- Для живых: `ages := ages + 1 + feed_bonus`; для мёртвых: `ages := 0`.

### 4.7 Сбор статистики без длинных списков

Вместо списков на все шаги используется:

- **онлайн-среднее и дисперсия** плотности (Welford) → `density_mean`, `stability`;
- **сумма и число** возрастов при гибели → `average_lifespan`;
- **deque фиксированной длины** (512 последних значений плотности) → `oscillation_score` (оценка автокорреляции по окну, а не по всей длине ряда);
- **одна копия** финального поля `life` → `diversity` через `pattern_diversity_from_frame`.

```mermaid
sequenceDiagram
  participant T as Шаг времени t
  participant L as life / food / ages
  T->>L: neighbors → birth/survival
  L->>L: noise flip
  L->>L: predation deaths
  L->>L: food regen + feeding
  L->>L: online death-age / density stats
```

---

## 5. Метрики: вектор \(M(\text{world}) \in \mathbb{R}^6\)

Метрики **`WorldMetrics`** вычисляются **внутри `run_world`** по онлайн-накопителям (см. §4.7). Отдельной функции `compute_metrics` нет.

| Имя | Как считается в коде | Пояснение |
|-----|----------------------|-----------|
| **`entropy`** | Бинарная энтропия Шеннона для **`density_mean`**: \(H(p)\) при \(p = \overline{\rho_t}\) | Не энтропия поля по паттернам, а энтропия «средней занятости во времени» как случайной бернуллиевской доли |
| **`stability`** | \(\mathrm{clip}(1 - \sigma(\rho)/(\mu(\rho)+\varepsilon), 0, 1)\) | Низкая дисперсия плотности во времени → выше стабильность |
| **`average_lifespan`** | `death_age_sum / death_count` (нет гибелей → `0`) | Среднее число шагов до гибели по умершим клеткам |
| **`density_mean`** | Онлайн-среднее плотности по шагам | Средняя заполненность поля живыми за прогон |
| **`oscillation_score`** | Автокорреляция по **окну** из последних 512 значений плотности | Приближение к «есть ли циклы» без хранения всего ряда |
| **`diversity`** | Доля уникальных подписей среди **`sample_size`** случайных патчей \(3\times3\) на **финальном** поле `life` | Грубая оценка «сколько разных локальных паттернов» |

Фиксированный порядок вектора задаётся методом **`WorldMetrics.as_vector()`**:

```text
[entropy, stability, average_lifespan, density_mean, oscillation_score, diversity]
```

```mermaid
flowchart LR
  subgraph traj["Поток по шагам"]
    DS["онлайн mean/var плотности"]
    WIN["окно плотности"]
    DA["сумма возрастов гибелей"]
    HA["финальный life"]
  end
  subgraph M["WorldMetrics"]
    e["entropy"]
    s["stability"]
    al["average_lifespan"]
    dm["density_mean"]
    os["oscillation_score"]
    dv["diversity"]
  end
  DS --> e
  DS --> s
  DS --> dm
  WIN --> os
  DA --> al
  HA --> dv
```

---

## 6. Пространство миров: PCA и кластеры

Функция **`stream_world_space_to_jsonl(generator, n_worlds, path, ...)`** (`src/worldspace/pipeline.py`):

1. **Проход 1:** для каждого мира из **`generator.iter_worlds(n)`** (без списка всех миров в памяти) — **`run_world`**, вектор метрик пишется в **временный memmap** `(n × 6)` и обновляются суммы для PCA (**`sum_x`**, **`sum_xx`**).
2. По достаточным статистикам: **`math.pca_mean_and_basis_2d`** → среднее и базис **2D**.
3. **k-means Lloyd** по строкам memmap (центроиды **`k×6`**, метки в отдельном memmap).
4. **Проход 2:** снова **`iter_worlds(n)`** (тот же порядок, детерминированные генераторы), чтение строки memmap, проекция **`math.project_pca_2d`**, запись **одной JSON-строки** в файл (и опционально в stdout).

Итоговая строка JSON содержит `world`, `metrics`, `embedding_2d`, `cluster_id`. Память по числу миров **не растёт** с размером батча (временный файл на диске для столбцов метрик).

```mermaid
flowchart TB
  subgraph batch["Батч из N миров"]
    W1["WorldSpec 1"]
    WN["WorldSpec N"]
  end
  subgraph metrics_mat["Матрица N×6"]
    MROW["каждая строка — as_vector(metrics)"]
  end
  subgraph proj["Проекция и группы"]
    PCA["PCA через SVD → (x,y)"]
    KM["k-means → cluster_id"]
  end
  W1 --> MROW
  WN --> MROW
  MROW --> PCA
  MROW --> KM
```

**Важно:** PCA здесь применяется **к метрикам поведения**, а не к «сырым» параметрам `WorldSpec`. Это создаёт карту «похожести поведения», которая может не совпадать с евклидовым расстоянием в пространстве правил.

---

## 7. Генераторы миров

Идея «лестницы генераторов»:

```mermaid
flowchart TB
  RW["random_walk(value)"]
  RWG["RandomWorldGenerator"]
  RWW["RandomWalkWorldGenerator"]
  MW["MarkovWorldGenerator"]
  TS["TwoStateNoiseMarkovGenerator"]
  RB["RuleBiasMarkovGenerator"]
  NN["NeuralWorldGenerator — заглушка"]
  LLM["LLMWorldGenerator — заглушка"]
  RW --> RWW
  RWG --> RWW
  MW --> TS
  MW --> RB
```

- **`RandomWorldGenerator`** — независимые случайные правила и параметры.
- **`RandomWalkWorldGenerator`** — последовательность миров: небольшие случайные изменения от стартового.
- **`TwoStateNoiseMarkovGenerator`** — скрытое состояние «спокойный / хаотичный» меняет масштаб шума.
- **`RuleBiasMarkovGenerator`** — смещение множеств `birth`/`survival`.
- **`NeuralWorldGenerator`**, **`LLMWorldGenerator`** — пока только **`NotImplementedError`** как место расширения.

---

## 8. CLI и файлы результатов

Запуск пакета как модуля использует **`src/worldspace/__main__.py`** → **`cli.main()`**.

Примеры:

```bash
python -m src.worldspace --generator random --worlds 30 --steps 200 --grid 40
```

Запись в файл — **JSONL** (одна JSON-строка на мир; память по числу миров **O(1)**):

```bash
python -m src.worldspace --output results/run.jsonl
```

Без **`--output`** те же строки идут в **stdout** (по одной JSON-строке на строку). Флаг **`--echo-lines`** дублирует строки в stdout при записи в файл.

Визуализация: **`src/worldspace/viz.py`**. **`--plot`** требует **`--output`**: график строится из JSONL через **`plot_world_embedding_from_jsonl`** (повторная симуляция не нужна). **`plot_simulation_final_grid`** использует **`result.final_life`**.

---

## 9. Ограничения и честные оговорки

1. **`cell_types`** и **`neighborhood`** в основном для совместимости со спецификацией «мира как JSON»; симулятор реализует **бинарную жизнь + еду + Moore**.
2. Начальная плотность жизни **фиксирована (20%)** в коде симулятора.
3. **`entropy`** — это не информационная энтропия конфигурации сетки, а функция от **средней плотности во времени**.
4. **`diversity`** — выборочная эвристика по последнему кадру, не полный спектр паттернов.
5. **Кластеризация и PCA** упрощённые (учебный MVP); для серьёзной карты миров имеет смысл нормализация признаков, другие методы снижения размерности и выбор расстояния.

---

## 10. Зависимости

Пакет использует **`numpy`**. Установка зависимостей проекта задаётся в **`pyproject.toml`**.

---

## См. также

- `src/worldspace/ARCHITECTURE.md` — короткая архитектурная выжимка на английском.
