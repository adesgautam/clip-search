
[![Build Status](https://travis-ci.com/adesgautam/clip-search.svg?branch=main)](https://travis-ci.com/adesgautam/clip-search)

## Clip-Search

## A search engine built using OpenAI's Clip model and FastAPI

Find the article for the code [here](https://adeshg7.medium.com/build-your-own-search-engine-using-openais-clip-and-fastapi-part-1-89995aefbcdd)

### Features
* Image Search
* Reverse Image Search
* Use `old_app.py` for brute force search
* Use `annoy_app.py` for nearest neighbours search using Spotify's Annoy

### Installation
To install the requirements use the following commands:
* `python3 -m pip install --upgrade pip`
* `python3 -m pip install torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/torch_stable.html`
* `python3 -m pip install -r requirements.txt`

### TODO
* Celery to run indexing job in background
* A better DB to store millions of records(Redis or PostgreSQL)
* PCA for reducing feature dimensions