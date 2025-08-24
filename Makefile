.PHONY: help build up down restart logs shell analyze clean

help:
	@echo "Available commands:"
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  make build    - Build Docker images"
	@echo "  make up       - Start all services"
	@echo "  make down     - Stop all services"
	@echo "  make restart  - Restart all services"
	@echo "  make logs     - View logs"
	@echo "  make shell    - Open shell in container"
	@echo "  make analyze  - Analyze a document"
	@echo "  make chat     - Start chat mode"
	@echo "  make clean    - Clean up volumes"
	@echo ""
	@echo "💻 Local Development Commands:"
	@echo "  make dev-setup     - Run local development setup"
	@echo "  make local-install - Install dependencies in venv"
	@echo "  make local-run     - Run status check locally"
	@echo "  make local-chat    - Start chat mode locally"
	@echo "  make local-analyze - Analyze document locally"
	@echo "  make local-summary - Show financial summary locally"
	@echo ""
	@echo "🔧 Utility Commands:"
	@echo "  make test     - Run tests"
	@echo "  make backup-db - Backup database"
	@echo "  make restore-db - Restore database"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 5
	docker-compose exec financial_agent python main.py status

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

shell:
	docker-compose exec financial_agent /bin/bash

analyze:
	@read -p "Enter document path: " doc_path; \
	docker-compose exec financial_agent python main.py analyze /app/documents/$$doc_path

chat:
	docker-compose exec -it financial_agent python main.py chat

summary:
	docker-compose exec financial_agent python main.py summary --period last_month

clean:
	docker-compose down -v
	rm -rf uploads/* exports/*

# Development commands
dev-setup:
	./setup-local.sh

local-install:
	source venv/bin/activate && pip install -r requirements.txt

local-run:
	source venv/bin/activate && python main.py status

local-chat:
	source venv/bin/activate && python main.py chat

local-analyze:
	@read -p "Enter document path: " doc_path; \
	source venv/bin/activate && python main.py analyze "$$doc_path"

local-summary:
	source venv/bin/activate && python main.py summary --period last_month

test:
	docker-compose exec financial_agent python -m pytest tests/

backup-db:
	docker-compose exec postgres pg_dump -U financial_user financial_agent > backup_$(shell date +%Y%m%d_%H%M%S).sql

restore-db:
	@read -p "Enter backup file: " backup_file; \
	docker-compose exec -T postgres psql -U financial_user financial_agent < $$backup_file
