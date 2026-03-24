#!/usr/bin/env python3
"""
SyncVoice 前端开发服务器
用法: python3 scripts/serve_frontend.py
访问: http://localhost:5500
"""
import http.server
import socketserver
import os

PORT = 5500
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Cache-Control", "no-store, no-cache")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # 静默日志

with socketserver.TCPServer(("", PORT), CORSHandler) as httpd:
    print(f"\n  前端服务器已启动")
    print(f"  URL: http://localhost:{PORT}")
    print(f"  目录: {ROOT}")
    print(f"\n  按 Ctrl+C 停止\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止")
