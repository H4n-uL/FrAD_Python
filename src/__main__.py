import sys
from .main import main
if __name__ == '__main__':
    sys.argv[0] = f'{sys.executable} -m {__package__}'
    main()
