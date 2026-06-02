# data_generation2.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Optional, Any, Dict

import numpy as np
import pandas as pd


@dataclass
class BTPConfig:
    n: int
    level_step: float = math.pi               # шаг уровня (по умолчанию π)
    base_level: float = 0.0                   # базовый уровень
    expected_seg_len: int = 200               # средняя длина сегмента (чем меньше, тем больше разладок)
    min_seg_len: int = 20                     # минимальная длина сегмента (чтобы не было "дроби")
    allow_random_walk: bool = False           # если True — уровни могут накапливаться вверх/вниз (0, π, 2π, ...)
    label_mode: str = "binary"                # "binary" | "signed" | "state"


class Binary_Telegraph_Process:
    """
    Генератор шумового ряда с разладками по уровню (mean shift).
    Варианты разметки:
      - label_mode="binary": labels_ ∈ {0,1}, 1 = смена уровня (бifurcation)
      - label_mode="signed": labels_ ∈ {-1,0,+1}, +1 вверх, -1 вниз, 0 нет разладки
                             + отдельная bifurcation_ ∈ {0,1}
      - label_mode="state": labels_ = индекс состояния (0/1/2/...) по времени
                            (полезно, если вы строите y_true через diff(state))
    """

    def __init__(
        self,
        n: int,
        noise_fn: Optional[Callable[..., Any]] = None,
        p1: Any = None,
        p2: Any = None,
        p3: Any = None,
        *,
        RANDOM_SEED: Optional[int] = None,
        alpha: Optional[float] = None,
        # параметры процесса уровней
        level_step: float = math.pi,
        base_level: float = 0.0,
        expected_seg_len: int = 200,
        min_seg_len: int = 20,
        allow_random_walk: bool = False,
        # разметка
        label_mode: str = "binary",
    ):
        self.cfg = BTPConfig(
            n=int(n),
            level_step=float(level_step),
            base_level=float(base_level),
            expected_seg_len=int(expected_seg_len),
            min_seg_len=int(min_seg_len),
            allow_random_walk=bool(allow_random_walk),
            label_mode=str(label_mode).lower().strip(),
        )

        self.RANDOM_SEED = RANDOM_SEED
        self._rng = np.random.default_rng(RANDOM_SEED)

        self.noise_fn = noise_fn
        self.p1, self.p2, self.p3 = p1, p2, p3
        self.alpha = alpha

        # --- генерируем процесс ---
        self._levels, self._state, self._direction, self._bkps, self._change_idx = self._generate_levels()
        self._noise = self._generate_noise()
        self._x = (self._noise + self._levels).astype(float)

        # --- метки ---
        self.labels_binary_ = np.zeros(self.cfg.n, dtype=int)
        self.labels_signed_ = np.zeros(self.cfg.n, dtype=int)
        if len(self._change_idx) > 0:
            self.labels_binary_[self._change_idx] = 1
            self.labels_signed_[self._change_idx] = self._direction[self._change_idx]

        # Это удобно отдельно иметь всегда:
        self.bifurcation_ = self.labels_binary_.copy()  # {0,1}
        self.direction_ = self.labels_signed_.copy()    # {-1,0,+1}

        # Главный "labels_" — как в вашем пайплайне
        if self.cfg.label_mode == "binary":
            self.labels_ = self.labels_binary_
        elif self.cfg.label_mode == "signed":
            self.labels_ = self.labels_signed_
        elif self.cfg.label_mode == "state":
            self.labels_ = self._state
        else:
            raise ValueError('label_mode must be one of: "binary", "signed", "state".')

    # ---------- public API (как в вашем ноутбуке) ----------

    def get_data(self):
        """Совместимо с вашим кодом: np.array(btp.get_data())"""
        return self._x

    def labels(self) -> pd.DataFrame:
        """
        Возвращает таблицу со всеми вариантами разметки и уровнем.
        """
        return pd.DataFrame(
            {
                "x": self._x,
                "level": self._levels,
                "state": self._state,
                "bifurcation": self.bifurcation_,
                "direction": self.direction_,
                "label": self.labels_,
            }
        )

    def bkps(self):
        """
        Список breakpoints в стиле ruptures: концы сегментов, включая n.
        (Может быть полезно для отладки / сравнения.)
        """
        return list(self._bkps)

    # ---------- internals ----------

    def _generate_levels(self):
        n = self.cfg.n
        # Длины сегментов ~ геометрическое распределение (дискретный аналог "пуассона по времени")
        p = 1.0 / max(1, self.cfg.expected_seg_len)

        bkps = []
        idx = 0
        while idx < n:
            seg_len = int(self._rng.geometric(p))
            seg_len = max(seg_len, self.cfg.min_seg_len)
            idx = min(n, idx + seg_len)
            bkps.append(idx)

        # Индексы начала новых сегментов (кроме 0)
        starts = [0] + bkps[:-1]
        change_idx = np.array(starts[1:], dtype=int)

        # Состояния и уровни
        state = np.zeros(n, dtype=int)
        levels = np.zeros(n, dtype=float)
        direction = np.zeros(n, dtype=int)  # {-1,0,+1} в момент смены

        current_state = 0
        current_level = self.cfg.base_level

        # для бинарного телеграфа: состояния 0 и 1
        # для random_walk: состояния накапливаются (0,1,2,...) а уровень = base + state*step
        prev_end = 0
        for seg_i, end in enumerate(bkps):
            # заполняем текущий сегмент
            state[prev_end:end] = current_state
            levels[prev_end:end] = current_level

            # подготовка к следующему сегменту
            if end < n:
                if self.cfg.allow_random_walk:
                    # случайный шаг вверх/вниз на один уровень
                    step = int(self._rng.choice([-1, +1]))
                    next_state = max(0, current_state + step)  # не уходим ниже 0
                    next_level = self.cfg.base_level + next_state * self.cfg.level_step
                else:
                    # бинарное переключение 0 <-> 1
                    next_state = 1 - current_state
                    next_level = self.cfg.base_level + next_state * self.cfg.level_step
                    step = +1 if next_state > current_state else -1

                # событие смены — ставим на индексе end (первый элемент нового сегмента)
                direction[end] = step

                current_state = next_state
                current_level = next_level

            prev_end = end

        return levels, state, direction, np.array(bkps, dtype=int), change_idx

    def _generate_noise(self) -> np.ndarray:
        n = self.cfg.n

        # Спец-режим "el_nino": либо загрузка локального файла, либо синтетика
        if (self.noise_fn is None) and (isinstance(self.p1, str)) and (self.p1.lower() == "el_nino"):
            return self._generate_el_nino_like(n)

        # По умолчанию — нормальный шум N(0,1)
        if self.noise_fn is None:
            return self._rng.normal(loc=0.0, scale=1.0, size=n)

        # Если пользователь передал np.random.normal — лучше использовать rng.normal (чтобы seed работал локально)
        if self.noise_fn is np.random.normal:
            loc = 0.0 if self.p1 is None else float(self.p1)
            scale = 1.0 if self.p2 is None else float(self.p2)
            return self._rng.normal(loc=loc, scale=scale, size=n)

        # Для colorednoise.powerlaw_psd_gaussian и похожих генераторов:
        # безопасно "подсеем" legacy RandomState, потому что многие библиотеки используют глобальный np.random
        old_state = None
        if self.RANDOM_SEED is not None:
            old_state = np.random.get_state()
            np.random.seed(self.RANDOM_SEED)

        try:
            # 1) пробуем сигнатуру (p1, size)
            try:
                noise = self.noise_fn(self.p1, n)
            except TypeError:
                # 2) пробуем (p1, p2, size=n) (например, некоторые кастомные генераторы)
                try:
                    noise = self.noise_fn(self.p1, self.p2, size=n)
                except TypeError:
                    # 3) последний шанс — (size=n) без параметров
                    noise = self.noise_fn(size=n)
        finally:
            if old_state is not None:
                np.random.set_state(old_state)

        noise = np.asarray(noise).reshape(-1)
        if noise.shape[0] != n:
            raise ValueError(f"noise_fn produced shape {noise.shape}, expected ({n},).")

        # alpha используем как масштаб амплитуды (как в вашем коде: alpha=1/2/pi)
        if self.alpha is not None:
            noise = noise * float(self.alpha)

        return noise.astype(float)

    def _generate_el_nino_like(self, n: int) -> np.ndarray:
        """
        Если у вас есть реальный ряд El Niño — положите рядом файл:
          - el_nino.csv с колонкой 'value' или первым числовым столбцом
        Иначе вернём синтетический "ENSO-like" сигнал (низкочастотная компонента + шум).
        """
        # 1) Попытка загрузки локального файла
        for fname in ("el_nino.csv", "el_nino.txt"):
            if os.path.exists(fname):
                df = pd.read_csv(fname)
                # ищем числовую колонку
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if not num_cols:
                    raise ValueError(f"{fname} найден, но не содержит числовых колонок.")
                x = df[num_cols[0]].to_numpy(dtype=float).reshape(-1)
                if len(x) < n:
                    # дополним циклическим повтором
                    reps = int(np.ceil(n / len(x)))
                    x = np.tile(x, reps)[:n]
                else:
                    x = x[:n]
                # нормализуем в разумный масштаб
                x = (x - x.mean()) / (x.std() + 1e-12)
                return x

        # 2) Синтетика: AR(1) + слабая сезонность
        phi = 0.995  # очень "красный" низкочастотный процесс
        eps = self._rng.normal(0.0, 1.0, size=n)
        x = np.zeros(n, dtype=float)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + eps[t]

        # добавим медленную синусоиду (условно)
        t = np.arange(n, dtype=float)
        season = np.sin(2.0 * np.pi * t / max(200.0, n / 20.0))
        x = x + 0.3 * season

        x = (x - x.mean()) / (x.std() + 1e-12)
        return x


# Удобный алиас (если где-то ожидается другое имя)
BTP = Binary_Telegraph_Process