"""Параметризованные расчёты для задачи №2."""
from __future__ import annotations
import math
from typing import Any, Iterable


def monthly_rate_from_annual(annual_rate: float) -> float:
    """Эффективная месячная ставка из эффективной годовой."""
    if annual_rate <= -1:
        raise ValueError("Годовая ставка должна быть больше -100%.")
    return (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0


def validate_config(cfg: dict[str, Any]) -> None:
    """Проверяет диапазоны и согласованность ключевых параметров."""
    positive = ["market_potential_users", "initial_users", "analysis_horizon_months"]
    non_negative = ["network_value_coefficient_rub_per_user_squared", "monthly_maintenance_cost_rub", "cac_rub", "arpu_rub_per_user_month"]
    for key in positive:
        if cfg[key] <= 0:
            raise ValueError(f"Параметр {key} должен быть положительным.")
    for key in non_negative:
        if cfg[key] < 0:
            raise ValueError(f"Параметр {key} не может быть отрицательным.")
    for key in ["growth_before_critical", "growth_after_critical", "annual_discount_rate", "monthly_discount_rate"]:
        if not 0 <= cfg[key] < 1:
            raise ValueError(f"Параметр {key} должен находиться в диапазоне [0; 1).")
    if cfg["initial_users"] > cfg["market_potential_users"]:
        raise ValueError("Начальная база не может превышать потенциал рынка.")
    cal = cfg["calendar"]
    if cal["analysis_horizon_months"] != cal["analysis_horizon_years"] * cal["months_per_year"]:
        raise ValueError("Горизонт в месяцах не согласован с горизонтом в годах.")
    if cfg["horizon_months"] != cfg["analysis_horizon_months"]:
        raise ValueError("horizon_months и analysis_horizon_months должны совпадать.")


def new_user_share(growth: float) -> float:
    """Доля новых пользователей в конечной базе при росте от предыдущей базы."""
    if growth <= -1:
        raise ValueError("Темп роста должен быть больше -100%.")
    return growth / (1.0 + growth)


def critical_mass(arpu: float, maintenance: float, cac: float, growth: float) -> float:
    """Минимальная база, при которой текущая прибыль не отрицательна.

    При N_t = N_(t-1)(1+g) фактический прирост равен N_t*g/(1+g).
    Поэтому прибыль = N_t*(ARPU-CAC*g/(1+g))-maintenance.
    """
    margin = arpu - cac * new_user_share(growth)
    if margin <= 0:
        return math.inf
    return maintenance / margin


def users_path(initial: float, growth_before: float, growth_after: float, critical: float, horizon: int, market_limit: float) -> list[float]:
    """Строит путь пользователей от месяца 0 до заданного горизонта."""
    users = [float(initial)]
    reached = initial >= critical
    for _ in range(horizon):
        rate = growth_after if reached else growth_before
        next_users = min(users[-1] * (1.0 + rate), market_limit)
        if not reached and next_users >= critical:
            reached = True
        users.append(next_users)
    return users


def months_to_target(target: float, growth: float, initial: float) -> float:
    if not math.isfinite(target):
        return math.inf
    if target <= initial:
        return 0
    if growth <= 0:
        return math.inf
    return math.ceil(math.log(target / initial) / math.log(1.0 + growth))


def cash_flow_table(users: Iterable[float], arpu: float, maintenance: float, cac: float) -> list[dict[str, float]]:
    """Формирует помесячные показатели; база месяца 0 уже существует."""
    values = list(users)
    rows = []
    for month, current in enumerate(values):
        previous = values[month - 1] if month else current
        new_users = 0.0 if month == 0 else max(current - previous, 0.0)
        revenue = arpu * current
        acquisition = cac * new_users
        profit = revenue - maintenance - acquisition
        rows.append({"month": month, "users": current, "new_users": new_users, "revenue": revenue, "maintenance": maintenance, "acquisition_cost": acquisition, "profit": profit})
    return rows


def npv(monthly_cash_flows: Iterable[float], rate: float) -> float:
    """NPV потоков по месяцам 1..T; месяц 0 не содержит инвестиционного потока."""
    return sum(float(cf) / ((1.0 + rate) ** month) for month, cf in enumerate(monthly_cash_flows, start=1))


def simulate(cfg: dict[str, Any], *, arpu: float | None = None, cac: float | None = None, growth_before: float | None = None, growth_after: float | None = None, maintenance: float | None = None) -> dict[str, Any]:
    """Полный пересчёт критической массы, траектории, прибыли и NPV."""
    arpu = cfg["arpu_rub_per_user_month"] if arpu is None else arpu
    cac = cfg["cac_rub"] if cac is None else cac
    growth_before = cfg["growth_before_critical"] if growth_before is None else growth_before
    growth_after = cfg["growth_after_critical"] if growth_after is None else growth_after
    maintenance = cfg["monthly_maintenance_cost_rub"] if maintenance is None else maintenance
    critical = critical_mass(arpu, maintenance, cac, growth_before)
    target = critical if math.isfinite(critical) else cfg["market_potential_users"] + 1
    users = users_path(cfg["initial_users"], growth_before, growth_after, target, cfg["analysis_horizon_months"], cfg["market_potential_users"])
    rows = cash_flow_table(users, arpu, maintenance, cac)
    rate = monthly_rate_from_annual(cfg["annual_discount_rate"]) if cfg["discounting"]["derive_monthly_from_annual"] else cfg["monthly_discount_rate"]
    for row in rows:
        row["discount_factor"] = 1.0 / ((1.0 + rate) ** row["month"])
        row["discounted_profit"] = row["profit"] * row["discount_factor"]
        row["network_value"] = cfg["network_value_coefficient_rub_per_user_squared"] * row["users"] ** 2
    return {"critical": critical, "users": users, "rows": rows, "npv": npv([r["profit"] for r in rows[1:]], rate), "monthly_rate": rate}


def first_positive_profit(rows: list[dict[str, float]]) -> int | None:
    for row in rows[1:]:
        if row["profit"] >= 0:
            return int(row["month"])
    return None


def binary_minimum(function, low: float, high: float, iterations: int = 70) -> float:
    """Минимальное x, для которого монотонная функция становится неотрицательной."""
    if function(high) < 0:
        return math.inf
    for _ in range(iterations):
        mid = (low + high) / 2.0
        if function(mid) >= 0:
            high = mid
        else:
            low = mid
    return high


def binary_maximum(function, low: float, high: float, iterations: int = 70) -> float:
    """Максимальное x, для которого функция остаётся неотрицательной."""
    if function(low) < 0:
        return -math.inf
    while function(high) >= 0 and high < 1e9:
        high *= 2
    if function(high) >= 0:
        return math.inf
    for _ in range(iterations):
        mid = (low + high) / 2.0
        if function(mid) >= 0:
            low = mid
        else:
            high = mid
    return low
