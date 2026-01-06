# 🧬 Evolve-Anything

**LLM-powered evolutionary optimization for arbitrary code.**

Evolve-Anything uses large language models to optimize code toward user-defined objectives. It combines the exploration power of evolutionary algorithms with the code understanding capabilities of modern LLMs.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

The code is based on [ShinkaEvolve](https://sakana.ai/shinka-evolve/) from Sakana AI.

It introduced several changes:
- Code refactoring and support for OpenRouter
- A more interactive Streamlit-based WebUI for exploring evolution history
- Multi-dimensional MAP-Elites
- Sandboxed code execution with [microsandbox](https://github.com/zerocore-ai/microsandbox)
- A research agent searching the web for new ideas and summarizing them.
