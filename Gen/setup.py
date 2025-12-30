from setuptools import setup, Extension

ext_modules = [
    Extension(
        "mldsa44",
        sources=["module.c", "mldsa44/polyvec.c", "mldsa44/poly.c", "mldsa44/reduce.c", "mldsa44/fips202.c", "mldsa44/rounding.c", "mldsa44/symmetric-shake.c", "mldsa44/ntt.c"],
        include_dirs=["mldsa44"],
        define_macros=[("MLDSA44", "1")],
    ),
    Extension(
        "mldsa65",
        sources=["module.c", "mldsa65/polyvec.c", "mldsa65/poly.c", "mldsa65/reduce.c", "mldsa65/fips202.c", "mldsa65/rounding.c", "mldsa65/symmetric-shake.c", "mldsa65/ntt.c"],
        include_dirs=["mldsa65"],
        define_macros=[("MLDSA65", "1")],
    ),
    Extension(
        "mldsa87",
        sources=["module.c", "mldsa87/polyvec.c", "mldsa87/poly.c", "mldsa87/reduce.c", "mldsa87/fips202.c", "mldsa87/rounding.c", "mldsa87/symmetric-shake.c", "mldsa87/ntt.c"],
        include_dirs=["mldsa87"],
        define_macros=[("MLDSA87", "1")],
    ),
]

setup(
    name="mldsa_py",
    version="0.1",
    ext_modules=ext_modules,
)
