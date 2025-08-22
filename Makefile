build:
	docker build --tag heart-risk .

run:
	docker rm -f heart-risk || true
	docker run --rm -it -p 8010:8000 -v $(PWD)/artifacts:/app/artifacts --name heart-risk heart-risk
