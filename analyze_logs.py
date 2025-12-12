#!/usr/bin/env python3
import json
import sys

stuck = [2,3,4,5,6,10,11,12,13,14,15,17,18,19,23]

with open('.cursor/debug.log') as f:
    logs = [json.loads(line) for line in f if line.strip()]

for iter in stuck:
    iter_logs = sorted([log for log in logs if log.get('data',{}).get('iteration') == iter], key=lambda x: x.get('timestamp',0))
    if iter_logs:
        last_log = iter_logs[-1]
        print(f'Iter {iter}: Last log = "{last_log.get("message","NONE")}" at timestamp {last_log.get("timestamp",0)}')
        # Show last 3 logs
        for log in iter_logs[-3:]:
            print(f'  -> {log.get("message","")}')



