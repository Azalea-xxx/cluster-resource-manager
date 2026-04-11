from flask import Flask, render_template, request, jsonify
import random
import numpy as np

app = Flask(__name__)

storage = {"nodes": [], "containers": []}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    req = request.json
    m, n = int(req.get('m', 5)), int(req.get('n', 10))
    mode = req.get('mode', 'random')

    if mode == 'random':
        # Популярные пресеты для рандома
        cpu_opts = [2, 4, 8, 16, 32, 64]
        ram_opts = [1024, 2048, 4096, 8192, 16384]
        storage['nodes'] = [{'id': j+1, 'cpu_max': random.choice(cpu_opts[2:]), 'ram_max': random.choice(ram_opts[2:])} for j in range(m)]
        storage['containers'] = [{'id': i+1, 'cpu': random.choice(cpu_opts[:3]), 'ram': random.choice(ram_opts[:3])} for i in range(n)]
    else:
        # Дефолтные значения для ручного ввода
        storage['nodes'] = [{'id': j+1, 'cpu_max': 16, 'ram_max': 4096} for j in range(m)]
        storage['containers'] = [{'id': i+1, 'cpu': 2, 'ram': 1024} for i in range(n)]
    
    return jsonify(storage)

@app.route('/update_data', methods=['POST'])
def update_data():
    global storage
    storage = request.json
    return jsonify({"status": "ok"})

# Функция Генетического алгоритма (ГА)
def run_genetic(nodes_data, containers_data, pop_size=50, generations=50):
    n, m = len(containers_data), len(nodes_data)
    population = [[random.randint(1, m) for _ in range(n)] for _ in range(pop_size)]

    def fitness(ind):
        cpu_u, ram_u = [0]*m, [0]*m
        penalty = 0
        for i, node_num in enumerate(ind):
            idx = node_num - 1
            cpu_u[idx] += containers_data[i]['cpu']
            ram_u[idx] += containers_data[i]['ram']
        for j in range(m):
            if cpu_u[j] > nodes_data[j]['cpu_max']: penalty += 100
            if ram_u[j] > nodes_data[j]['ram_max']: penalty += 100
        return len(set(ind)) + penalty

    for _ in range(generations):
        population = sorted(population, key=fitness)
        next_gen = population[:10]
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(population[:15], 2)
            child = p1[:n//2] + p2[n//2:]
            if random.random() < 0.1: child[random.randint(0, n-1)] = random.randint(1, m)
            next_gen.append(child)
        population = next_gen
    return sorted(population, key=fitness)[0]

@app.route('/solve', methods=['POST'])
def solve():
    method = int(request.json.get('method', 1))
    nodes = [dict(node, cpu_used=0, ram_used=0, items=[]) for node in storage['nodes']]
    containers = storage['containers']
    unplaced = 0

    if method == 4:
        best_r = run_genetic(storage['nodes'], storage['containers'])
        for i, node_num in enumerate(best_r):
            j = node_num - 1
            if nodes[j]['cpu_used'] + containers[i]['cpu'] <= nodes[j]['cpu_max'] and nodes[j]['ram_used'] + containers[i]['ram'] <= nodes[j]['ram_max']:
                nodes[j]['cpu_used'] += containers[i]['cpu']
                nodes[j]['ram_used'] += containers[i]['ram']
                nodes[j]['items'].append(containers[i]['id'])
            else: unplaced += 1
    else:
        for i, c in enumerate(containers):
            target = -1
            if method == 1: # First Fit
                for j in range(len(nodes)):
                    if nodes[j]['cpu_used'] + c['cpu'] <= nodes[j]['cpu_max'] and nodes[j]['ram_used'] + c['ram'] <= nodes[j]['ram_max']:
                        target = j; break
            elif method == 2: # Best Fit
                best_val = float('inf')
                for j in range(len(nodes)):
                    if nodes[j]['cpu_used'] + c['cpu'] <= nodes[j]['cpu_max'] and nodes[j]['ram_used'] + c['ram'] <= nodes[j]['ram_max']:
                        val = nodes[j]['cpu_max'] - (nodes[j]['cpu_used'] + c['cpu'])
                        if val < best_val: best_val = val; target = j
            elif method == 3: # Worst Fit
                worst_val = -1
                for j in range(len(nodes)):
                    if nodes[j]['cpu_used'] + c['cpu'] <= nodes[j]['cpu_max'] and nodes[j]['ram_used'] + c['ram'] <= nodes[j]['ram_max']:
                        val = nodes[j]['cpu_max'] - (nodes[j]['cpu_used'] + c['cpu'])
                        if val > worst_val: worst_val = val; target = j
            if target != -1:
                nodes[target]['cpu_used'] += c['cpu']; nodes[target]['ram_used'] += c['ram']
                nodes[target]['items'].append(c['id'])
            else: unplaced += 1

    n_used = sum(1 for n in nodes if n['items'])
    cpu_l = [(n['cpu_used']/n['cpu_max']*100) if n['cpu_max']>0 else 0 for n in nodes]
    ram_l = [(n['ram_used']/n['ram_max']*100) if n['ram_max']>0 else 0 for n in nodes]
    avg_c, avg_r = np.mean(cpu_l), np.mean(ram_l)
    disp = np.mean([((cpu_l[j]-avg_c)**2 + (ram_l[j]-avg_r)**2) for j in range(len(nodes))])

    return jsonify({
        "nodes": nodes,
        "metrics": {"n_used": n_used, "p_unplaced": round(unplaced/len(containers)*100, 2), "avg_cpu": round(avg_c, 2), "avg_ram": round(avg_r, 2), "dispersion": round(disp, 2)}
    })

if __name__ == '__main__':
    app.run(debug=True)