import os, sys

def check(root: str) -> int:
    bad = []
    for dp, _, files in os.walk(root):
        for f in files:
            if f.endswith('.py'):
                p = os.path.join(dp, f)
                try:
                    with open(p, 'rb') as fh:
                        src = fh.read()
                    compile(src, p, 'exec')
                except Exception as e:
                    bad.append((p, e))
    if bad:
        print('Syntax errors:')
        for p, e in bad:
            print(' -', p, '\n   ', repr(e))
        return 1
    print('All Python files compiled successfully.')
    return 0

if __name__ == '__main__':
    sys.exit(check(sys.argv[1] if len(sys.argv) > 1 else 'azimuth-fm'))

