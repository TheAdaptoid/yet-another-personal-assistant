# Lint the package
lint:
    uvx ruff check

# Fix lint errors within the package
lint-fix:
    uvx ruff check --fix

# Typecheck the package
typecheck:
    uvx ty check

# Test the package
test:
    uv run pytest