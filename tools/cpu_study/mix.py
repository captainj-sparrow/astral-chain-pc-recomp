"""Instruction mix of one lifted function in an object file.

usage: mix.py <object> <symbol>
"""
import collections
import re
import subprocess
import sys

s = subprocess.run(['C:/msys64/ucrt64/bin/objdump.exe', '-d', '-M', 'intel', '--no-show-raw-insn', f'--disassemble={sys.argv[2]}',
                    sys.argv[1]], capture_output=True, text=True).stdout
ins = [l.split('\t', 1)[1].strip() for l in s.splitlines() if re.match(r'^\s+[0-9a-f]+:\t', l)]
c = collections.Counter()
for l in ins:
    op = l.split()[0]
    mem = re.search(r'\[([a-z0-9]+)[^\]]*\]', l)
    base = mem.group(1) if mem else None
    if op.startswith('j'):
        k = 'branch'
    elif op in ('call', 'ret', 'push', 'pop'):
        k = 'call/ret/push/pop'
    elif base == 'rsp':
        k = 'host stack ' + ('store' if re.search(r'PTR \[[^\]]*\],', l) else 'load/use')
    elif mem and re.search(r'PTR \[[^\]]*\],', l):
        k = 'store (Cpu or guest memory)'
    elif op in ('vucomiss', 'vucomisd', 'vcomiss', 'vcomisd', 'vtestps', 'vptest', 'vtestpd') or op.startswith('vcmp'):
        k = 'fp compare/test'
    elif op in ('vandps', 'vandpd', 'vandnps', 'vpand'):
        k = 'fp and (abs)'
    elif mem:
        k = 'load/use memory'
    elif op.startswith('v'):
        k = 'vector/fp op'
    elif op.startswith('set') or op.startswith('cmov') or op in ('cmp', 'test', 'bt'):
        k = 'int compare/set/cmov'
    else:
        k = 'int op'
    c[k] += 1
print(f'{sys.argv[2]}: {len(ins)} instructions')
for k, n in c.most_common():
    print(f'  {n:5d} {100 * n / len(ins):5.1f}%  {k}')
