version = "1.0"

train {
    step train {
        image = "basisai/workload-standard:v0.1.2"
        install = [
            "pip3 install --upgrade pip",
            "pip3 install -r requirements.txt",
        ]
        script = [{sh = ["python3 train.py"]}]
        resources {
            cpu = "2"
            memory = "12G"
        }
    }
}
