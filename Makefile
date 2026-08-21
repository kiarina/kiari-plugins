.PHONY: list update upgrade format lint check
.DEFAULT_GOAL := check
#--------------------------------------------------
list:
	uv pip list
update:
	uv sync --all-extras --all-groups
	uv pip list --outdated
upgrade:
	uv sync --upgrade --all-extras --all-groups
#--------------------------------------------------
format:
	mise run format
lint:
	mise run lint
#--------------------------------------------------
check:
	mise run format
	mise run lint
