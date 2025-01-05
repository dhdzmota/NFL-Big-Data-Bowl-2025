import subprocess

command_list = [
    'python src/compute_features.py',
    'python src/train_model.py',
]

for command in command_list:
    subprocess.run(command.split(' '))
