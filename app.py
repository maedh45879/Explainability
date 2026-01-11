import warnings

warnings.filterwarnings("ignore", message=".*np\\.object.*", category=FutureWarning)

from src.ui.pages import build_app


def main():
    demo = build_app()
    #demo.launch()
    demo.launch(debug=True)


if __name__ == "__main__":
    main()
