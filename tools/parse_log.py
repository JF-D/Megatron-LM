import os


def read_log(filename):
    speeds, memory = [], 0
    with open(filename, 'r') as fp:
        for line in fp.readlines():
            if 'max allocated' in line:
                memory = max(memory, float(line.split('|')[2].split()[-1]))
            elif 'iteration' in line:
                speed = float(line.split('|')[2].split()[-1])
                speeds.append(speed)
            elif line.startswith('Speed:'):
                speed = float(line.split()[-1].split('m')[0])
                speeds.append(speed)

    return min(speeds) if len(speeds) > 0 else 0, memory

if __name__ == '__main__':
    stats = {}
    for filename in os.listdir('log'):
        if not filename.endswith('.log'):
            continue
        model, ps, ndev = filename[:-4].split('_')
        ndev = int(ndev)
        if model not in stats:
            stats[model] = {}
        if ps not in stats[model]:
            stats[model][ps] = {}
        speed, memory = read_log(f'log/{filename}')
        stats[model][ps][ndev] = (speed, memory)

    for model, model_stats in stats.items():
        for ps, ps_stats in model_stats.items():
            print(f'Model: {model}, PS: {ps}')
            print(f'ndev, speed(ms/iter),  memory(MB)')
            for ndev in sorted(list(ps_stats.keys())):
                print(f'{ndev:3}, {ps_stats[ndev][0]:.3f}, {ps_stats[ndev][1]:.3f}')
