def Run_Line():
    import subprocess
    import os

    subprocess.run(["ls", "-l"])

    # for complex commands, with many args, use string + `shell=True`:
    cmd_str = "darknet detector test vehicles.data vehicles.cfg vehicles_best.weights Hacktech2023.jpg"

    subprocess.run(cmd_str, shell=True)
    os.remove("Hacktech2023.jpg")