from setuptools.command.build_ext import build_ext
from pathlib import Path
import subprocess
import os
from setuptools import Extension, setup


class BuildWithPreprocessing(build_ext):
    def run(self):
        # Run the preprocessing step first
        self.run_preprocessing()

        # Make sure numpy include dirs are added **after numpy is installed**
        self.add_numpy_include()

        # Continue with normal build_ext
        super().run()

    def add_numpy_include(self):
        try:
            import numpy as np
        except ImportError:
            raise RuntimeError(
                "NumPy must be installed before building this extension."
            )

        for ext in self.extensions:
            if np.get_include() not in ext.include_dirs:
                ext.include_dirs.append(np.get_include())

    def run_preprocessing(self):
        root = Path(__file__).parent.absolute()

        cc = os.environ.get("CC", "gcc")
        cflags = os.environ.get("CFLAGS", "-O2").split()

        compile_cmd = [
            cc,
            *cflags,
            "-o",
            str(root / "precalculate_table"),
            str(root / "source/precalculate_table.c"),
            str(root / "source/elliptic_integral.c"),
        ]

        print("Running preprocessing step...")

        try:
            subprocess.run(compile_cmd, check=True, cwd=root)
            print("✓ Compilation successful")

            subprocess.run([str(root / "precalculate_table")], check=True, cwd=root)
            print("✓ Preprocessing completed")

            target_dir = Path(self.build_lib) / "Finite"
            print(root, target_dir)
            subprocess.run(
                ["mv", str(root / "func0.dat"), str(target_dir)],
                check=True,
                cwd=root,
            )
            subprocess.run(
                ["mv", str(root / "func1.dat"), str(target_dir)],
                check=True,
                cwd=root,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error during preprocessing: {e}")
            raise


ext_modules = [
    Extension(
        name="Finite_C",
        sources=["source/python_3_wrapper.c", "source/finite.c"],
        include_dirs=[],
    )
]

setup(
    name="Finite",
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildWithPreprocessing,
    },
    setup_requires=["numpy"],
)
