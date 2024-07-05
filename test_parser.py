import argparse

def main():
    parser = argparse.ArgumentParser(description='Test parser for hydropower and transmission options.')
    parser.add_argument('--hydro', type=str, choices=['y', 'n'], default='n', help='Include hydropower: y/n')
    parser.add_argument('--transmission', type=str, choices=['y', 'n'], default='n', help='Include transmission: y/n')
    args = parser.parse_args()

    include_hydro = args.hydro == 'y'
    include_transmission = args.transmission == 'y'

    print(f"Include hydropower: {include_hydro}")
    print(f"Include transmission: {include_transmission}")

if __name__ == "__main__":
    main()
