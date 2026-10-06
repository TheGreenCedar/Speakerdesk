"""Record actual hosted runtime capability, without model weights or capture."""
import argparse
import json
from pathlib import Path
import subprocess

def main(runtime,output):
    result=subprocess.run([str(runtime),'--model-capability'],check=True,text=True,capture_output=True,timeout=15)
    report=json.loads(result.stdout)
    if report.get('scope')!='capability_only' or report.get('frozen') is not True or report.get('models_executed') is not False:
        raise ValueError('Unexpected capability scope; this cannot satisfy acoustic acceptance')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.runtime,args.output)
