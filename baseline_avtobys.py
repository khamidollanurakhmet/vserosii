# -*- coding: utf-8 -*-
"""

- Пример запуска: `python baseline_avtobys.py --mode heuristic --n_stops 30`
- Сохранение решения в JSON: `--save solution_avtobys.json`
"""
from __future__ import annotations

import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

try:
    import networkx as nx
except Exception:
    nx = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# Опционально: ILP/CP-SAT
try:
    import pulp
except Exception:
    pulp = None

try:
    from ortools.sat.python import cp_model
except Exception:
    cp_model = None

# ---------------------- Конфиг ----------------------
@dataclass
class AvtoConfig:
    seed: int = 42
    buses_total: int = 20
    capacity: int = 50
    max_route_len: float = 10.0  # км (условных)
    max_stops_per_route: int = 12
    max_routes: int = 8
    sa_iters: int = 2000
    sa_start_temp: float = 1.0
    sa_end_temp: float = 0.01
    unserved_penalty: float = 5.0
    route_km_cost: float = 1.0
    save_dir: str = 'solutions'

CFG = AvtoConfig()

# ---------------------- Утилиты ----------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------- Данные задачи ----------------------
@dataclass
class Problem:
    coords: np.ndarray  # (N,2) координаты остановок
    demand: np.ndarray  # (N,N) матрица спроса OD (симм/несимм)
    dist: np.ndarray    # (N,N) матрица расстояний (км)
    depot: int          # индекс депо (опционально используем)
    buses_total: int
    capacity: int
    max_route_len: float
    max_stops_per_route: int
    max_routes: int
    unserved_penalty: float
    route_km_cost: float


@dataclass
class Route:
    stops: List[int]        # последовательность остановок, замкнутая или линейная
    buses_assigned: int     # сколько автобусов выделено (>=1)


@dataclass
class Solution:
    routes: List[Route]

    def to_json(self) -> Dict[str, Any]:
        return {
            'routes': [
                {
                    'stops': r.stops,
                    'buses_assigned': r.buses_assigned,
                } for r in self.routes
            ]
        }


# ---------------------- Генерация синтетики ----------------------

def generate_synthetic_problem(n_stops: int = 30, seed: int = 42, buses_total: int = 20, capacity: int = 50) -> Problem:
    set_seed(seed)
    coords = np.random.rand(n_stops, 2) * 10.0
    # расстояния — евклид
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    # спрос — пропорционален близости центров плотности
    base = np.random.gamma(shape=2.0, scale=2.0, size=n_stops)
    demand = np.outer(base, base)
    # занулить диагональ
    np.fill_diagonal(demand, 0.0)
    # нормировать и масштабировать
    demand = demand / demand.mean() * 5.0
    depot = int(np.argmin(coords.sum(axis=1)))
    return Problem(
        coords=coords,
        demand=demand,
        dist=dist,
        depot=depot,
        buses_total=buses_total,
        capacity=capacity,
        max_route_len=CFG.max_route_len,
        max_stops_per_route=CFG.max_stops_per_route,
        max_routes=CFG.max_routes,
        unserved_penalty=CFG.unserved_penalty,
        route_km_cost=CFG.route_km_cost,
    )


# ---------------------- Оценка решения ----------------------

def route_length(route: List[int], dist: np.ndarray, closed: bool = True) -> float:
    if len(route) < 2:
        return 0.0
    length = 0.0
    for i in range(len(route) - 1):
        length += dist[route[i], route[i+1]]
    if closed:
        length += dist[route[-1], route[0]]
    return float(length)


def served_mask_by_routes(routes: List[List[int]], n: int) -> np.ndarray:
    """Булева матрица обслуживания OD: 1 если i и j на одном маршруте."""
    served = np.zeros((n, n), dtype=bool)
    for r in routes:
        rset = set(r)
        idx = np.array(sorted(list(rset)))
        if idx.size == 0:
            continue
        # отметим все пары внутри множества маршрута
        for i in idx:
            served[i, idx] = True
            served[idx, i] = True
        served[idx, idx] = False
    return served


def evaluate_solution(problem: Problem, sol: Solution) -> Dict[str, float]:
    n = problem.demand.shape[0]
    routes = [r.stops for r in sol.routes if len(r.stops) > 1 and r.buses_assigned > 0]
    served = served_mask_by_routes(routes, n)
    served_demand = float(problem.demand[served].sum())

    # штраф за необслуженный спрос
    total_demand = float(problem.demand.sum())
    unserved = total_demand - served_demand
    penalty = unserved * problem.unserved_penalty

    # стоимость пробега
    km_cost = 0.0
    for r in sol.routes:
        km = route_length(r.stops, problem.dist)
        km_cost += km * max(0, r.buses_assigned)

    # ограничение на автобусы
    buses_used = sum(max(0, r.buses_assigned) for r in sol.routes)
    buses_violation = max(0, buses_used - problem.buses_total)
    buses_penalty = buses_violation * 1e6

    # ограничение на длину маршрута
    len_violation = 0
    for r in sol.routes:
        if route_length(r.stops, problem.dist) > problem.max_route_len:
            len_violation += 1
    len_penalty = len_violation * 1e6

    # целевая: минимизировать total_cost
    total_cost = penalty + km_cost * problem.route_km_cost + buses_penalty + len_penalty

    return {
        'served_demand': served_demand,
        'total_demand': total_demand,
        'unserved': unserved,
        'penalty_unserved': penalty,
        'km_cost': km_cost,
        'buses_used': float(buses_used),
        'len_violations': float(len_violation),
        'objective': float(total_cost),
    }


# ---------------------- Начальная конструкция решения ----------------------

def greedy_hub_routes(problem: Problem) -> Solution:
    n = problem.demand.shape[0]
    # важность остановки — суммарный спрос к/от нее
    importance = problem.demand.sum(axis=0) + problem.demand.sum(axis=1)
    hubs = list(np.argsort(-importance)[: max(2, min(problem.max_routes, n))])

    routes: List[Route] = []
    for h in hubs:
        # построим короткую звезду вокруг хаба (соседние остановки по близости)
        d = problem.dist[h].copy()
        order = list(np.argsort(d))
        path = [h]
        for v in order:
            if v == h:
                continue
            if len(path) >= problem.max_stops_per_route:
                break
            trial = path + [v]
            if route_length(trial, problem.dist) <= problem.max_route_len:
                path = trial
        # замкнём цикл к депо, если выгодно
        if problem.depot not in path:
            trial = path + [problem.depot]
            if route_length(trial, problem.dist) <= problem.max_route_len:
                path = trial
        routes.append(Route(stops=path, buses_assigned=1))

    # распределим автобусы оставшиеся равномерно
    residual = max(0, problem.buses_total - len(routes))
    i = 0
    while residual > 0 and len(routes) > 0:
        routes[i % len(routes)].buses_assigned += 1
        residual -= 1
        i += 1

    return Solution(routes=routes)


# ---------------------- Локальный поиск: симулированный отжиг ----------------------

def sa_temperature(t: int, t_max: int, t0: float, t1: float) -> float:
    alpha = t / max(1, t_max)
    return t0 * (t1 / max(1e-8, t0)) ** alpha


def perturb(problem: Problem, sol: Solution) -> Solution:
    # операции: swap в маршруте, move между маршрутами, add/remove остановку
    new_sol = Solution(routes=[Route(stops=list(r.stops), buses_assigned=r.buses_assigned) for r in sol.routes])
    if not new_sol.routes:
        return new_sol
    op = random.choice(['swap', 'move', 'add', 'remove', 'buses'])
    if op == 'swap':
        r = random.choice(new_sol.routes)
        if len(r.stops) >= 2:
            i, j = sorted(random.sample(range(len(r.stops)), 2))
            r.stops[i], r.stops[j] = r.stops[j], r.stops[i]
    elif op == 'move' and len(new_sol.routes) >= 2:
        r1, r2 = random.sample(new_sol.routes, 2)
        if r1.stops:
            idx = random.randrange(len(r1.stops))
            node = r1.stops.pop(idx)
            pos = random.randrange(0, len(r2.stops)+1)
            r2.stops.insert(pos, node)
    elif op == 'add':
        r = random.choice(new_sol.routes)
        # попробуем добавить случайную остановку, близкую к последней
        candidates = list(range(problem.demand.shape[0]))
        random.shuffle(candidates)
        for v in candidates:
            if v in r.stops:
                continue
            pos = random.randrange(0, len(r.stops)+1)
            trial = r.stops[:pos] + [v] + r.stops[pos:]
            if len(trial) <= problem.max_stops_per_route and route_length(trial, problem.dist) <= problem.max_route_len:
                r.stops = trial
                break
    elif op == 'remove':
        r = random.choice(new_sol.routes)
        if len(r.stops) > 2:
            idx = random.randrange(len(r.stops))
            r.stops.pop(idx)
    elif op == 'buses':
        # перекинуть один автобус между маршрутами
        if len(new_sol.routes) >= 2:
            rich = [r for r in new_sol.routes if r.buses_assigned > 1]
            if rich:
                r1 = random.choice(rich)
                r2 = random.choice(new_sol.routes)
                if r1 is not r2:
                    r1.buses_assigned -= 1
                    r2.buses_assigned += 1
    return new_sol


def simulated_annealing(problem: Problem, init: Solution, iters: int = None) -> Tuple[Solution, Dict[str, float]]:
    if iters is None:
        iters = CFG.sa_iters
    cur = init
    cur_score = evaluate_solution(problem, cur)
    best = cur
    best_score = cur_score
    for t in tqdm(range(iters), desc='SA'):
        T = sa_temperature(t, iters, CFG.sa_start_temp, CFG.sa_end_temp)
        cand = perturb(problem, cur)
        cand_score = evaluate_solution(problem, cand)
        delta = cand_score['objective'] - cur_score['objective']
        if delta < 0 or random.random() < math.exp(-delta / max(1e-8, T)):
            cur, cur_score = cand, cand_score
            if cand_score['objective'] < best_score['objective']:
                best, best_score = cand, cand_score
    return best, best_score


# ---------------------- ILP (опционально, простая модель покрытия) ----------------------

def ilp_cover_routes(problem: Problem, candidates: List[List[int]]) -> Tuple[Optional[Solution], Optional[Dict[str, float]]]:
    if pulp is None:
        return None, None
    n = problem.demand.shape[0]
    # предвычислим покрытие
    cover = []
    for r in candidates:
        mask = served_mask_by_routes([r], n)
        cover.append(mask)
    # переменные выбора маршрутов (0/1)
    model = pulp.LpProblem('route_cover', pulp.LpMinimize)
    x = [pulp.LpVariable(f'x_{i}', lowBound=0, upBound=1, cat='Binary') for i in range(len(candidates))]
    b = [pulp.LpVariable(f'b_{i}', lowBound=0, cat='Integer') for i in range(len(candidates))]

    # покрытие спроса
    served_pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # если хотя бы один маршрут покрывает пару — обслужено
            # линейная аппроксимация: s_ij <= sum x_r * cover[r][i,j]
            s_ij = pulp.lpSum(x[r] for r in range(len(candidates)) if cover[r][i, j])
            served_pairs.append((i, j, s_ij))

    # бюджет автобусов
    model += pulp.lpSum(b) <= problem.buses_total

    # длина маршрутов
    for i, r in enumerate(candidates):
        model += route_length(r, problem.dist) <= problem.max_route_len + 1e-6

    # связь автобусов и выбора: если маршрут выбран, на нём >=1 автобус
    for i in range(len(candidates)):
        model += b[i] >= x[i]

    # цель: штраф за необслуженный спрос + стоимость пробега
    total_demand = float(problem.demand.sum())
    served_sum = pulp.lpSum(problem.demand[i, j] * pulp.lpSum([x[r] for r in range(len(candidates)) if cover[r][i, j]]) for i in range(n) for j in range(n) if i != j)
    penalty = (total_demand - served_sum) * problem.unserved_penalty
    km_cost = pulp.lpSum(route_length(candidates[i], problem.dist) * b[i] * problem.route_km_cost for i in range(len(candidates)))
    model += penalty + km_cost

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    x_val = [int(v.value() >= 0.5) for v in x]
    b_val = [int(round(max(0.0, (bv.value() or 0.0)))) for bv in b]
    routes = []
    for i, choose in enumerate(x_val):
        if choose and b_val[i] > 0:
            routes.append(Route(stops=candidates[i], buses_assigned=b_val[i]))
    sol = Solution(routes=routes)
    score = evaluate_solution(problem, sol)
    return sol, score


def candidate_routes_greedy(problem: Problem, k: int = 20) -> List[List[int]]:
    n = problem.demand.shape[0]
    imp = problem.demand.sum(axis=0) + problem.demand.sum(axis=1)
    centers = list(np.argsort(-imp)[: max(2, min(n, problem.max_routes * 2))])
    cands = []
    for c in centers:
        order = list(np.argsort(problem.dist[c]))
        path = [c]
        for v in order:
            if v == c:
                continue
            trial = path + [v]
            if len(trial) <= problem.max_stops_per_route and route_length(trial, problem.dist) <= problem.max_route_len:
                path = trial
        cands.append(path)
    # небольшие вариации
    for _ in range(max(0, k - len(cands))):
        base = random.choice(cands)
        if len(base) > 2:
            i, j = sorted(random.sample(range(len(base)), 2))
            cand = base[:]
            cand[i], cand[j] = cand[j], cand[i]
            if route_length(cand, problem.dist) <= problem.max_route_len:
                cands.append(cand)
    # уникальные по множеству остановок
    uniq = []
    seen = set()
    for r in cands:
        key = tuple(sorted(set(r)))
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


# ---------------------- IO ----------------------

def save_solution_json(sol: Solution, path: str) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sol.to_json(), f, ensure_ascii=False, indent=2)


def load_problem_json(path: str) -> Problem:
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    coords = np.array(obj['coords'], dtype=float)
    demand = np.array(obj['demand'], dtype=float)
    dist = np.array(obj['dist'], dtype=float) if 'dist' in obj else np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    return Problem(
        coords=coords,
        demand=demand,
        dist=dist,
        depot=int(obj.get('depot', 0)),
        buses_total=int(obj.get('buses_total', CFG.buses_total)),
        capacity=int(obj.get('capacity', CFG.capacity)),
        max_route_len=float(obj.get('max_route_len', CFG.max_route_len)),
        max_stops_per_route=int(obj.get('max_stops_per_route', CFG.max_stops_per_route)),
        max_routes=int(obj.get('max_routes', CFG.max_routes)),
        unserved_penalty=float(obj.get('unserved_penalty', CFG.unserved_penalty)),
        route_km_cost=float(obj.get('route_km_cost', CFG.route_km_cost)),
    )



# ---------------------- CLI ----------------------

def parse_args():
    p = argparse.ArgumentParser(description='Генеративный бейзлайн оптимизации автобусных потоков')
    p.add_argument('--mode', type=str, default='heuristic', choices=['heuristic', 'sa', 'ilp'], help='решатель: эвристика/SA/ILP')
    p.add_argument('--n_stops', type=int, default=30, help='число остановок для синтетики')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--load', type=str, default=None, help='путь к JSON с задачей')
    p.add_argument('--save', type=str, default=None, help='куда сохранить решение JSON')
    p.add_argument('--iters', type=int, default=None, help='число итераций для SA')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.load:
        problem = load_problem_json(args.load)
    else:
        problem = generate_synthetic_problem(n_stops=args.n_stops, seed=args.seed)

    if args.mode == 'heuristic':
        init = greedy_hub_routes(problem)
        sol, score = init, evaluate_solution(problem, init)
    elif args.mode == 'sa':
        init = greedy_hub_routes(problem)
        sol, score = simulated_annealing(problem, init, iters=args.iters)
    elif args.mode == 'ilp':
        cands = candidate_routes_greedy(problem, k=30)
        sol, score = ilp_cover_routes(problem, cands)
        if sol is None:
            print('ILP требования не установлены (pulp). Запускаю эвристику.')
            init = greedy_hub_routes(problem)
            sol, score = init, evaluate_solution(problem, init)
    else:
        raise ValueError('Неизвестный режим')

    print('Результаты:')
    for k, v in score.items():
        print(f'{k}: {v:.4f}')

    if args.save:
        save_solution_json(sol, args.save)
        print(f'Сохранено решение в {args.save}')


if __name__ == '__main__':
    main()
