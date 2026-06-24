from flask import Flask, render_template, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
import random
import numpy as np
import time
import copy
import logging
import os

LOG_DIR = "/var/log/sim"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/sim.log"),
        logging.StreamHandler()  # для вывода в консоль
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

metrics = PrometheusMetrics(app)

from prometheus_client import Counter, Histogram, Gauge

algorithm_requests = Counter('sim_algorithm_requests_total', 'Total requests by algorithm', ['algorithm'])

algorithm_duration = Histogram('sim_algorithm_duration_seconds', 'Duration of algorithm execution', ['algorithm'])

unplaced_gauge = Gauge('sim_unplaced_containers', 'Unplaced containers percentage')

nodes_used_gauge = Gauge('sim_nodes_used', 'Number of used nodes')

active_requests = Gauge('sim_active_requests', 'Active requests')
 
metrics.info('app_info', 'SRE Simulator Info', version='1.0.0')

storage = {"nodes": [], "containers": []}

def genetic_algorithm(nodes_data, containers_data, pop_size=20, generations=50):
    num_containers = len(containers_data)
    num_nodes = len(nodes_data)
    
    def get_fitness(individual):
        # individual — это список, где индекс = контейнер, значение = узел
        temp_nodes = [copy.deepcopy(n) for n in nodes_data]
        for n in temp_nodes:
            n.update({'cpu_used': 0, 'ram_used': 0, 'items': []})
            
        unplaced = 0
        for c_idx, n_idx in enumerate(individual):
            c = containers_data[c_idx]
            n = temp_nodes[n_idx]
            if n['cpu_used'] + c['cpu'] <= n['cpu_max'] and n['ram_used'] + c['ram'] <= n['ram_max']:
                n['cpu_used'] += c['cpu']
                n['ram_used'] += c['ram']
                n['items'].append(c['id'])
            else:
                unplaced += 1
        
        n_used = sum(1 for n in temp_nodes if n['items'])
        # Оценка: чем меньше узлов и чем меньше неразмещенных — тем лучше
        # Добавляем очень большой штраф за неразмещенные
        return n_used + (unplaced * 100), temp_nodes, unplaced

    # 1. Начальная популяция
    population = [[random.randint(0, num_nodes - 1) for _ in range(num_containers)] for _ in range(pop_size)]
    
    best_ind = None
    best_fit = float('inf')
    best_nodes = []
    best_unplaced = 0

    for _ in range(generations):
        # Оценка
        fits = []
        for ind in population:
            f, _, _ = get_fitness(ind)
            fits.append(f)
            
        # Отбор 
        sorted_pop = [x for _, x in sorted(zip(fits, population))]
        population = sorted_pop[:pop_size//2] # Оставляем 50%
        
        # Скрещивание и мутация 
        while len(population) < pop_size:
            parent = random.choice(sorted_pop[:5])
            child = list(parent)
            # Мутация
            child[random.randint(0, num_containers-1)] = random.randint(0, num_nodes-1)
            population.append(child)
            
        # Сохраняем лучшее решение
        current_best_fit, current_nodes, current_unplaced = get_fitness(sorted_pop[0])
        if current_best_fit < best_fit:
            best_fit = current_best_fit
            best_nodes = current_nodes
            best_unplaced = current_unplaced

    return best_nodes, best_unplaced

# --- ОБЩАЯ ЛОГИКА РАЗМЕЩЕНИЯ ---
def run_placement_logic(method_id, nodes_data, containers_data):
    nodes = [dict(n, cpu_used=0, ram_used=0, items=[]) for n in nodes_data]
    unplaced = 0

    if method_id in [1, 2, 3]:
        for c in containers_data:
            target = -1
            if method_id == 1: # First Fit
                for j in range(len(nodes)):
                    if nodes[j]['cpu_used'] + c['cpu'] <= nodes[j]['cpu_max'] and nodes[j]['ram_used'] + c['ram'] <= nodes[j]['ram_max']:
                        target = j; break
            elif method_id == 2: # Best Fit (минимум остатка CPU)
                best_val = float('inf')
                for j in range(len(nodes)):
                    if nodes[j]['cpu_used'] + c['cpu'] <= nodes[j]['cpu_max'] and nodes[j]['ram_used'] + c['ram'] <= nodes[j]['ram_max']:
                        val = nodes[j]['cpu_max'] - (nodes[j]['cpu_used'] + c['cpu'])
                        if val < best_val: best_val = val; target = j
            elif method_id == 3: # Worst Fit (максимум остатка CPU)
                worst_val = -1
                for j in range(len(nodes)):
                    if nodes[j]['cpu_used'] + c['cpu'] <= nodes[j]['cpu_max'] and nodes[j]['ram_used'] + c['ram'] <= nodes[j]['ram_max']:
                        val = nodes[j]['cpu_max'] - (nodes[j]['cpu_used'] + c['cpu'])
                        if val > worst_val: worst_val = val; target = j
            
            if target != -1:
                nodes[target]['cpu_used'] += c['cpu']
                nodes[target]['ram_used'] += c['ram']
                nodes[target]['items'].append(c['id'])
            else:
                unplaced += 1
    
    elif method_id == 4: # ГЕНЕТИЧЕСКИЙ АЛГОРИТМ
        nodes, unplaced = genetic_algorithm(nodes_data, containers_data)

    # Расчет метрик
    n_used = sum(1 for n in nodes if n['items'])
    cpu_loads = [(n['cpu_used']/n['cpu_max']*100) if n['cpu_max']>0 else 0 for n in nodes]
    ram_loads = [(n['ram_used']/n['ram_max']*100) if n['ram_max']>0 else 0 for n in nodes]
    
    avg_cpu = np.mean(cpu_loads) if cpu_loads else 0
    avg_ram = np.mean(ram_loads) if ram_loads else 0
    disp = np.mean([((l - avg_cpu)**2 + (r - avg_ram)**2) for l, r in zip(cpu_loads, ram_loads)]) if nodes else 0
    
    return {
        "nodes": nodes,
        "metrics": {
            "n_used": n_used,
            "p_unplaced": round(unplaced/len(containers_data)*100, 2) if containers_data else 0,
            "avg_cpu": round(avg_cpu, 2),
            "avg_ram": round(avg_ram, 2),
            "dispersion": round(disp, 2)
        }
    }

@app.route('/')
def index(): return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    req = request.json
    m, n = int(req.get('m', 5)), int(req.get('n', 10))
    storage['nodes'] = [{'id': j+1, 'cpu_max': 16, 'ram_max': 4096} for j in range(m)]
    storage['containers'] = [{'id': i+1, 'cpu': 2, 'ram': 512} for i in range(n)]
    return jsonify(storage)

@app.route('/update_data', methods=['POST'])
def update_data():
    global storage
    storage = request.json
    return jsonify({"status": "ok"})

@app.route('/solve', methods=['POST'])
def solve():
    m_id = int(request.json.get('method', 1))
    algorithm_name = ['First Fit', 'Best Fit', 'Worst Fit', 'Genetic'][m_id - 1]

    logger.info(f"Запуск алгоритма {algorithm_name} (ID={m_id})")

    algorithm_requests.labels(algorithm=algorithm_name).inc()

    start = time.perf_counter()
    res = run_placement_logic(m_id, storage['nodes'], storage['containers'])
    duration = time.perf_counter() - start

    algorithm_duration.labels(algorithm=algorithm_name).observe(duration)

    unplaced_gauge.set(res['metrics']['p_unplaced'])
    nodes_used_gauge.set(res['metrics']['n_used'])

    res['metrics']['exec_time'] = round(duration * 1000, 3)

    logger.info(f"Алгоритм {algorithm_name} завершён за {res['metrics']['exec_time']} мс")

    return jsonify(res)

@app.route('/compare_all', methods=['POST'])
def compare_all():
    summary = []
    for m_id in range(1, 5):
        start = time.perf_counter()
        res = run_placement_logic(m_id, storage['nodes'], storage['containers'])
        summary.append({
            "name": ["First Fit", "Best Fit", "Worst Fit", "Genetic"][m_id-1],
            "metrics": res['metrics'],
            "time": round((time.perf_counter() - start) * 1000, 3)
        })
    return jsonify(summary)


if __name__ == '__main__':
    app.run(debug=True, port=5000)