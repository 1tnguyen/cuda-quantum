
from flask import Flask, jsonify, request, Response
import fire
from typing import Union, List
import pathlib
import os
import json
import requests
import time
import sys

class Server:
    def __init__(self, port: int = 3030):
        self.api = Flask(__name__)
        self.port = port

        @self.api.route('/job', methods=["POST"])
        def run():
            qpud_up = False
            retries = 0
            while (not qpud_up):
                try:
                    ping_reponse = requests.get('http://localhost:3031/') 
                    qpud_up = (ping_reponse.status_code == 200)
                except:
                    qpud_up = False
                if not qpud_up:
                    retries+=1
                    if retries > 100:
                        sys.exit()
                    print("Main application is down, retrying (retry_count = {})...".format(retries))
                    time.sleep(0.1) 
            
            request_headers = {k:v for k,v in request.headers}
            for k in list(request_headers):
                if k.lower().startswith("nvcf"):
                    request_headers[k.upper()] = request_headers[k]
            res = requests.request(
                method = request.method,
                url = request.url.replace(request.host_url, "http://localhost:3031/"),
                headers = request_headers,
                data = request.get_data(),
                cookies = request.cookies,
                allow_redirects = False,
            )

            return res.json()

        @self.api.route('/', methods=["GET"])
        def status():
            return {"status": "OK"}

    def run(self):
        self.api.run(debug=False, host="0.0.0.0", port=self.port)

if __name__ == '__main__':
    fire.Fire(Server)

