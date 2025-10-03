from fastapi import FastAPI
app=FastAPI()
@app.get('/status')
def s(): return {'ok':True,'db':False}
