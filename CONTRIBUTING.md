# Contributing to Zonos API

We welcome bug reports, feature ideas, fixes, and new maintainers.

## Zonos model repository

This API imports the `zonos` package from a separate repository. This project has no git submodule.

1. Clone Zonos into the root of this repository.
```bash
git clone https://github.com/Zyphra/Zonos.git
```

2. Install it in editable mode from the repository root.
```bash
pip install -e Zonos
```

3. Update Zonos when you need a newer version.
```bash
cd Zonos
git pull
cd ..
```
Record the commit you tested against in the pull request description when the change depends on a specific Zonos version.

## Development process

1. Clone this repository and set up the environment.
```bash
git clone https://github.com/manascb1344/zonos-api
cd zonos-api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/Zyphra/Zonos.git
pip install -e Zonos
pip install -e .
```

2. Create a branch.
```bash
git checkout -b feature/your-feature-name
```

3. Make your changes and run the application.
```bash
uvicorn app.main:app --reload
```

4. Run the tests.
```bash
pytest
```

5. Update the documentation.
- Update docstrings for changed functions.
- Update `README.md` when endpoints or request fields change.
- Update API documentation when routes change.

6. Commit your changes using [Conventional Commits](https://www.conventionalcommits.org/).
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `chore:` maintenance tasks
- `refactor:` code changes that preserve behavior

## Docker development

```bash
docker-compose up --build
docker-compose logs -f zonos
```

## Pull request process

1. Fork the repository and create your branch from `main`.
2. Update the documentation.
3. Describe what you tested and the Zonos commit you used.
4. Open the pull request.

## License

By contributing, you agree that your contributions are licensed under the Apache 2.0 License.
