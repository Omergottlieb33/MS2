"""ms2 <command> [options]

    run        one recording, CZI to per-cell activity      (ms2.pipeline)
    compare    several recordings, then their comparison    (ms2.pipeline_compare)
    view       masks / tracks / ROI viewer in the browser   (ms2.viewer.app)
    dashboard  tracking-quality dashboard in the browser    (ms2.tracking_dashboard)

ms2 <command> -h lists that command's options.
"""
import importlib
import sys

COMMANDS = {
    'run': 'ms2.pipeline',
    'compare': 'ms2.pipeline_compare',
    'view': 'ms2.viewer.app',
    'dashboard': 'ms2.tracking_dashboard',
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help') else 2)
    command = sys.argv.pop(1)
    sys.argv[0] = f'ms2 {command}'   # so the command's --help shows how it was called
    importlib.import_module(COMMANDS[command]).main()


if __name__ == '__main__':
    main()
