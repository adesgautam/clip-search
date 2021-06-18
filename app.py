
import sqlite3
import glob
import torch
import clip
import uvicorn
import numpy as np
import pandas as pd
import shutil

from fastapi import FastAPI
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, File, UploadFile

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

CONTENT_STORE = 'static'

DATABASE = 'database.db'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print("Loading Model...")

def create_tables():
	conn = sqlite3.connect(DATABASE)
	conn.execute('CREATE TABLE IF NOT EXISTS image_embeds (id INTEGER PRIMARY KEY, image_path TEXT, embedding BLOB)')
	# conn.execute('CREATE TABLE IF NOT EXISTS text_embeds (id TEXT, text TEXT, embedding TEXT)')
	conn.commit()
	print('image_embeds table created')

def preprocess_image(image_path):
	image = Image.open(image_path)
	image = preprocess(image).unsqueeze(0).to(device)
	return image

def preprocess_text(text):
	return clip.tokenize(text).to(device)

def create_image_embeddings(image_paths):
	conn = sqlite3.connect(DATABASE)
	for img_path in image_paths:
		processed_image = preprocess_image(img_path)
		with torch.no_grad():
			embed = model.encode_image(processed_image).detach().numpy()
		embed = embed.tostring()

		## Insert data into sqlite3
		query = "INSERT INTO image_embeds(image_path, embedding) VALUES(?, ?);"
		conn.execute(query, (img_path, embed))
		conn.commit()
		print('Inserted', img_path)
	return 'success'

def cal_sim(feat1, feat2):
	img_embed = np.fromstring(feat2, dtype=np.float32)
	img_embed = img_embed.reshape((1, img_embed.shape[0]))
	sim = cosine_similarity(feat1, img_embed)
	return sim[0][0]

def text_images_similarity(text, df):
	## Preprocess text
	processed_text = preprocess_text([text])
	with torch.no_grad():
		text_embed = model.encode_text(processed_text)
	
	## Calculate cos sim for all images wrt to text
	df['sim'] = df['embedding'].apply(lambda x: cal_sim(text_embed, x))
	
	df = df.sort_values(by=['sim'], ascending=False)
	return df

def image_images_similarity(img_path, df):
	## Preprocess image
	processed_image = preprocess_image(img_path)
	with torch.no_grad():
		image_embed = model.encode_image(processed_image).detach().numpy()

	df['sim'] = df['embedding'].apply(lambda x: cal_sim(image_embed, x))

	df = df.sort_values(by=['sim'], ascending=False)
	return df
	
def get_image_data():
	conn = sqlite3.connect(DATABASE)
	curr = conn.cursor()

	query = "SELECT image_path, embedding FROM image_embeds;"
	curr.execute(query)

	rows = curr.fetchall()
	return rows

def get_image_data_df():
	con = sqlite3.connect(DATABASE)
	df = pd.read_sql("SELECT image_path, embedding FROM image_embeds;", con)
	return df

## Create tables
create_tables()

@app.get("/")
def index():
	return {"message": "Hello World!"}

@app.get("/create_image_embeds")
def create_image_embeds():
	image_paths = glob.glob(CONTENT_STORE+'/*.jpg') + glob.glob(CONTENT_STORE+'/*.jpeg')
	print(image_paths)
	status = create_image_embeddings(image_paths)
	if status=='success':
		return "Inserted"
	else:
		return "Failed"

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

@app.post("/imagesearch", response_class=HTMLResponse)
async def imagesearch(request: Request):
	form_data = await request.form()
	text = form_data['text']

	## Get whole data from DB
	df = get_image_data_df()
	df_sim = text_images_similarity(text, df)
	
	images = df_sim['image_path'].tolist()

	context = {"request": request, "images": images, "text": text}
	return templates.TemplateResponse("index.html", context)

@app.post("/revimagesearch", response_class=HTMLResponse)
async def revimagesearch(request: Request, uploaded_file: UploadFile = File(...)):
	file_location = f"upload/{uploaded_file.filename}"
	with open(file_location, "wb+") as file_object:
		file_object.write(uploaded_file.file.read())

	print('Saved at file_location', file_location)

	## Copy uploaded file to static dir
	static_file_location = file_location.replace('upload/', 'static/')
	shutil.copy(file_location, static_file_location)

	df = get_image_data_df()
	df_sim = image_images_similarity(file_location, df)

	images = df_sim['image_path'].tolist()

	context = {"request": request, 'images': images, "query_img": static_file_location}
	return templates.TemplateResponse("reverse_image_search.html", context)

# @app.post("/upload-file/")
# async def create_upload_file(uploaded_file: UploadFile = File(...)):
#     file_location = f"upload/{uploaded_file.filename}"
#     with open(file_location, "wb+") as file_object:
#         file_object.write(uploaded_file.file.read())
#     return {"info": f"file '{uploaded_file.filename}' saved at '{file_location}'"}

if __name__ == "__main__":
	uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)














