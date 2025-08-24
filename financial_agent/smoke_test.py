from __future__ import annotations

from .agent import run_once


def main() -> None:
    print(run_once("List my recent transactions and advise on saving."))


if __name__ == "__main__":
    main()
