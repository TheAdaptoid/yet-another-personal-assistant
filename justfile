# Lint the package
lint:
    uvx ruff check

# Typecheck the package
typecheck:
    uvx ty check

# Test the package
test:
    uv run pytest