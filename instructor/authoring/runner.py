"""Source-aware learner CLI; always grades learner code, never the instructor reference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

def catalog():
    return json.loads((ROOT / 'course/stages.json').read_text())

def fingerprint():
    paths = [ROOT/'pyproject.toml', ROOT/'uv.lock', ROOT/'course/stages.json']
    for folder in ['src','grader','course_cli']:
        paths.extend(sorted((ROOT/folder).rglob('*.py')))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

def check(args, stages):
    selected = stages[args.stage-1:args.stage] if args.only else stages[:args.stage]
    tests = list(dict.fromkeys(t for s in selected for t in s['tests']))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    xml = ROOT / '.course/runs' / f'{stamp}.xml'
    xml.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable,'-m','pytest','grader/test_boundary.py',*tests,
               '--implementation', 'toyvllm' if (ROOT/'src/toyvllm').exists() else 'learner.api',
               '--case-seed',str(args.seed),'-q','--tb=short','--maxfail=1',
               '-p','pytest_timeout',f'--junitxml={xml}']
    env = os.environ.copy()
    for key in ['PYTEST_ADDOPTS','PYTEST_PLUGINS']:
        env.pop(key,None)
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(ROOT)
    before = fingerprint()
    print(f"Stage {args.stage}: {stages[args.stage-1]['title']} — {'isolated debug' if args.only else 'cumulative gate'}",flush=True)
    try:
        code = subprocess.run(command,cwd=ROOT,env=env,timeout=300).returncode
    except subprocess.TimeoutExpired:
        print('Grading exceeded 300 seconds; no completion recorded.')
        code=1
    counts={'passed':0,'failed':0,'skipped':0}
    if xml.exists():
        for case in ET.parse(xml).iter('testcase'):
            kind='failed' if case.find('failure') is not None or case.find('error') is not None else 'skipped' if case.find('skipped') is not None else 'passed'
            counts[kind]+=1
    good=code==0 and counts['passed']>1 and not counts['failed'] and not counts['skipped'] and before==fingerprint()
    record=dict(stage=args.stage,timestamp=stamp,fingerprint=before,seed=args.seed,complete=good and not args.only,selection='isolated' if args.only else 'cumulative',exit_code=code,junit=str(xml.relative_to(ROOT)),runtime=dict(python=platform.python_version(),platform=platform.platform()),**counts)
    (ROOT/'.course/latest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'Verified stages 1–{args.stage}.' if record['complete'] else 'No cumulative completion claim. Read the stage contract or request a conceptual hint.')
    return 0 if good else (code or 1)

def main():
    p=argparse.ArgumentParser(description='Read, implement, verify. Learner-owned code; instructor-owned tests.')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list'); sub.add_parser('status')
    for name in ['read','hint','check']:
        q=sub.add_parser(name); q.add_argument('stage',type=int)
        if name=='hint': q.add_argument('--level',type=int,choices=[1,2,3],default=1)
        if name=='check':
            q.add_argument('--only',action='store_true'); q.add_argument('--seed',type=int,default=1729)
    args=p.parse_args(); stages=catalog()
    if hasattr(args,'stage') and not 1<=args.stage<=len(stages):
        p.error(f'stage must be between 1 and {len(stages)}')
    if args.command=='list':
        for s in stages: print(f"{s['number']:02} {s['title']}\n   implement: {s['symbols']}")
    elif args.command=='read':
        stage=stages[args.stage-1]
        print((ROOT/stage['chapter']).read_text())
        if stage.get('textbook'): print('\n---\n'+(ROOT/stage['textbook']).read_text())
    elif args.command=='hint': print(stages[args.stage-1]['hints'][args.level-1])
    elif args.command=='check': return check(args,stages)
    else:
        path=ROOT/'.course/latest.json'
        if not path.exists(): print('No grading run yet. Start: uv run course check 1')
        else:
            r=json.loads(path.read_text()); fresh=r['fingerprint']==fingerprint()
            print(f"Latest stage {r['stage']}: {r['passed']} passed, {r['failed']} failed, {r['skipped']} skipped")
            print('Source unchanged.' if fresh else 'STALE: implementation, grader or environment changed.')
            print(f"Verified through stage {r['stage']}." if fresh and r['complete'] else 'No current cumulative completion claim.')
    return 0

if __name__=='__main__': raise SystemExit(main())
