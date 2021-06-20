
import sqlite3
import glob
import torch
import clip
import uvicorn
import numpy as np
import pandas as pd
import random
import string

from fastapi import FastAPI
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, File, UploadFile
from fastapi import BackgroundTasks

from annoy import AnnoyIndex

from typing import List
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

CONTENT_STORE = 'static'
DATABASE = 'database.db'
ANNOY_INDEX_FILE = 'annoy_indexes.ann'
NUM_OF_RESULTS_TO_SHOW = 5
NUM_OF_TREES_TO_BUILD = 5
INDEX_METRIC = 'angular'

embed_length = 512
annoy_indexer = AnnoyIndex(embed_length, INDEX_METRIC)

ANNOY_INDEX = 0

print("Loading Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def create_tables():
	conn = sqlite3.connect(DATABASE)
	conn.execute('CREATE TABLE IF NOT EXISTS image_embeds (id INTEGER PRIMARY KEY, annoy_index TEXT, image_path TEXT, embedding BLOB)')
	conn.execute('CREATE TABLE IF NOT EXISTS text_embeds (id INTEGER PRIMARY KEY, text_seq TEXT, embedding BLOB)')
	conn.commit()

def preprocess_image(image_path):
	image = Image.open(image_path)
	image = preprocess(image).unsqueeze(0).to(device)
	return image

def preprocess_text(text):
	return clip.tokenize(text).to(device)

def store_uploaded_images(uploaded_files):
	image_paths = []
	for uploaded_file in uploaded_files:
		rand_post_text = ''.join(random.choices(string.ascii_uppercase+string.digits, k = 10))
		fname_split = uploaded_file.filename.split('.')
		fname = fname_split[0] + rand_post_text + "." + fname_split[1]
		file_location = f"{CONTENT_STORE}/{fname}"

		with open(file_location, "wb+") as file_object:
			file_object.write(uploaded_file.file.read())

		image_paths.append(file_location)
	return image_paths

def create_embedding(conn, image_path):
	global ANNOY_INDEX

	processed_image = preprocess_image(image_path)
	with torch.no_grad():
		embed = model.encode_image(processed_image).detach().numpy()

	embed_str = embed.tostring()
	## Insert data into DB
	query = "INSERT INTO image_embeds(annoy_index, image_path, embedding) VALUES(?, ?, ?);"
	conn.execute(query, (ANNOY_INDEX, image_path, embed_str))
	conn.commit()

	annoy_indexer.add_item(ANNOY_INDEX, embed[0])

	ANNOY_INDEX += 1
	print('Inserted', image_path)

def create_image_embeddings(uploaded_files):
	conn = sqlite3.connect(DATABASE)
	image_paths = store_uploaded_images(uploaded_files)

	for image_path in image_paths:
		create_embedding(conn, image_path)

	annoy_indexer.build(NUM_OF_TREES_TO_BUILD)
	annoy_indexer.save(ANNOY_INDEX_FILE)

	update_annoy_index()
	return 'Embeddings Created !'

def get_image_data_df():
	con = sqlite3.connect(DATABASE)
	df = pd.read_sql("SELECT image_path, embedding FROM image_embeds;", con)
	return df

def exec_query(query):
	conn = sqlite3.connect(DATABASE)
	curr = conn.cursor()
	curr.execute(query)
	rows = curr.fetchall()
	return rows

def update_annoy_index():
	global ANNOY_INDEX
	query = "SELECT id FROM image_embeds ORDER BY id DESC LIMIT 1;"
	rows = exec_query(query)
	if rows==[]:
		ANNOY_INDEX = 0
	else:
		ANNOY_INDEX = int(rows[0][0]) + 1
	print('Updated Annoy Index as:', ANNOY_INDEX)

## GET APIs
@app.get("/")
def index():
	return {"message": "Hello World!"}

@app.get("/startindexinghome", response_class=HTMLResponse)
def imagesearchhome(request: Request):
	return templates.TemplateResponse("create_img_embeds.html", {"request": request})

@app.get("/imagesearchhome", response_class=HTMLResponse)
def imagesearchhome(request: Request):
	df = get_image_data_df()
	images = df['image_path'].tolist()
	return templates.TemplateResponse("index.html", {"request": request, "images": images})

@app.get("/revimagesearchhome", response_class=HTMLResponse)
def revimagesearchhome(request: Request):
	df = get_image_data_df()
	images = df['image_path'].tolist()
	return templates.TemplateResponse("reverse_image_search.html", {"request": request, "images": images})

## POST APIs
@app.post("/startindexing", response_class=HTMLResponse)
async def create_image_embeds(request: Request, background_tasks: BackgroundTasks, 
								uploaded_files: List[UploadFile] = File(...)):
	background_tasks.add_task(create_image_embeddings, uploaded_files)
	return templates.TemplateResponse("create_img_embeds.html", {"request": request, "msg": "Indexing Started"})

@app.post("/imagesearch", response_class=HTMLResponse)
async def imagesearch(request: Request):
	global NUM_OF_RESULTS_TO_SHOW

	form_data = await request.form()
	text = form_data['text']

	## Using Annoy indexer
	ann_indexer = AnnoyIndex(embed_length, INDEX_METRIC)
	ann_indexer.load(ANNOY_INDEX_FILE)
	total_indexes = ann_indexer.get_n_items()
	print('Total Indexes in Annoy Indexer:', ann_indexer.get_n_items())

	## Get Text Embedding
	processed_text = preprocess_text([text])
	with torch.no_grad():
		text_embed = model.encode_text(processed_text)

	if NUM_OF_RESULTS_TO_SHOW > total_indexes:
		NUM_OF_RESULTS_TO_SHOW = total_indexes

	## Get Indexes
	indexes = ann_indexer.get_nns_by_vector(text_embed[0], NUM_OF_RESULTS_TO_SHOW, search_k=-1, include_distances=False)
	indexes = [str(i) for i in indexes]
	print('Indexes found relevant to the query:', indexes)

	images = []
	for index in indexes:
		query = "SELECT image_path from image_embeds WHERE annoy_index = {0}".format(index)
		row = exec_query(query)
		images.append(row[0][0])

	print(images)	

	context = {"request": request, "images": images, "text": text}
	return templates.TemplateResponse("index.html", context)

@app.post("/revimagesearch", response_class=HTMLResponse)
async def revimagesearch(request: Request, uploaded_file: UploadFile = File(...)):
	global NUM_OF_RESULTS_TO_SHOW

	rand_post_text = ''.join(random.choices(string.ascii_uppercase+string.digits, k = 10))
	fname_split = uploaded_file.filename.split('.')
	fname = fname_split[0] + rand_post_text + "." + fname_split[1]
	file_location = f"{CONTENT_STORE}/{fname}"

	with open(file_location, "wb+") as file_object:
		file_object.write(uploaded_file.file.read())

	print('Saved at file_location', file_location)

	## Using Annoy indexer
	ann_indexer = AnnoyIndex(embed_length, INDEX_METRIC)
	ann_indexer.load(ANNOY_INDEX_FILE)
	total_indexes = ann_indexer.get_n_items()
	print('Total Indexes in Annoy Indexer:', ann_indexer.get_n_items())

	## Get Text Embedding
	processed_text = preprocess_image(file_location)
	with torch.no_grad():
		text_embed = model.encode_text(processed_text)

	if NUM_OF_RESULTS_TO_SHOW > total_indexes:
		NUM_OF_RESULTS_TO_SHOW = total_indexes

	## Get Indexes
	indexes = ann_indexer.get_nns_by_vector(text_embed[0], NUM_OF_RESULTS_TO_SHOW, search_k=-1, include_distances=False)
	indexes = [str(i) for i in indexes]
	print('Indexes found relevant to the query:', indexes)

	query = "SELECT image_path from image_embeds WHERE annoy_index IN ({0})".format(','.join(indexes))
	rows = exec_query(query)

	images = [i[0] for i in rows]
	print(images)	

	context = {"request": request, 'images': images, "query_img": file_location}
	return templates.TemplateResponse("reverse_image_search.html", context)

## Create tables
create_tables()
update_annoy_index()

if __name__ == "__main__":
	uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)


