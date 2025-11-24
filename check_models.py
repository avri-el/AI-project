from google import genai

client = genai.Client(api_key="AIzaSyCVEO-gFt9AElVBCZ5BpALNF5DmlOOA_3Q")

models = client.models.list()

for m in models:
    print(m.name)
