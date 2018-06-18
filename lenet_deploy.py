import sys
import requests
data = {
    "command": "python lenet5_caffe.py " + " ".join(sys.argv[1:]),
    "image": "rombh/summer1_pytorch1:latest",
    "arch": "PYTORCH",
    "gpus_required": 1,
    "user": "summer1",
    "apikey": "summer1"
}
# print data
# print join(sys.argv[:]

r = requests.post(' http://52.168.107.28:5000/jobs', json=data)
job_id = r.json()['coreid']
URL = 'http://52.168.107.28:5000/jobs/' + job_id

print URL

while True:
    x = raw_input('fetch ?')
    if x:
        break
    r2 = requests.get(URL)
    x = r2.json()
    print x['last_stdout'].strip().replace('\n\n', '\n') + "\n============="
    if x['status'] == 'SHUTDOWN':
        break
