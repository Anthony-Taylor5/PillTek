from flask import Flask, request
import subprocess

app = Flask(__name__)

running_process = None   


@app.route("/trigger", methods=["POST"])
def trigger():
    global running_process

    event = request.json.get("event")
    print(f"ESP32 event: {event}")

    if event == "beacon_near":

        # Only start if not already running
        if running_process is None or running_process.poll() is not None:
            print("Starting subprocesses...")
            running_process = subprocess.Popen(["python", "detect_people.py"])
        else:
            print("subprocesses already running")

        return "ok"

    elif event == "beacon_far":

        # Stop the running process if active
        if running_process is not None and running_process.poll() is None:
            print("Stopping subprocesses...")
            running_process.terminate()

            try:
                running_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                print("Process did not terminate, force killing...")
                running_process.kill()

        running_process = None
        return "ok"

    else:   
        return "invalid event", 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
