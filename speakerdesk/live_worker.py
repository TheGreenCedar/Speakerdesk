"""Offline resident-worker entry point; growing provisional tail and rolling refinement."""
import json
import sys

def emit(message):
    print(json.dumps(message,ensure_ascii=False),flush=True)

def run(config):
    from live_refinement import run as resident
    return resident(config,emit)

if __name__=='__main__':
    try:run(json.loads(sys.argv[1]))
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit({'type':'error','error':str(error)});sys.exit(1)
