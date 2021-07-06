import spacy
import sys
from collections import defaultdict
from flask import Flask,render_template,url_for,request
import re
import pandas as pd

nlp = spacy.load('en')

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/process',methods=["POST"])
def process():
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		with nlp.disable_pipes('ner'):
			doc = nlp(rawtext)
		beams = nlp.entity.beam_parse([ doc ], beam_width = 16, beam_density = 0.0001)
		entity_scores = defaultdict(float)
		for beam in beams:
			for score, ents in nlp.entity.moves.get_beam_parses(beam):
				for start, end, label in ents:
					entity_scores[(start, end, label)] += score
		print ('Entities and scores (detected with beam search)')
		results = []
		for key in entity_scores:
			start, end, label = key
			score = entity_scores[key]
			if (label=='ORG'):
				results.append([label, doc[start:end], score])
				print ('Label: {}, Name: {}, Score: {}'.format(label, doc[start:end], score))


	return render_template("index.html",results=results)


if __name__ == '__main__':
	app.run(debug=True)