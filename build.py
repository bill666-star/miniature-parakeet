import PyInstaller.__main__
import sys, os

def build():
    rdkit = os.path.join(sys.base_prefix, "Lib", "site-packages", "rdkit")
    PyInstaller.__main__.run([
        "skelgen_pro_v2.py",
        "--onefile",
        "--windowed",
        "--noconsole",
        "--name=SkelGen_Pro_v2.0",
        f"--add-data={rdkit};rdkit",
        "--clean"
    ])

if __name__ == "__main__":
    build()
