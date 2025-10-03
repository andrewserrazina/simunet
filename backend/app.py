from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

app=FastAPI()
@app.get('/status')
def s(): return {'ok':True,'db':False}

@app.get("/", include_in_schema=False)
def home():
    db_state = "online" if DB_AVAILABLE else "offline"
    return HTMLResponse(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SimuNet API</title>
<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:2rem;color:#eaf2ff;background:#0b1020}
a{color:#7bb1ff;text-decoration:none} .box{padding:1rem;border:1px solid #273469;border-radius:12px;background:#0f1430;max-width:680px}
code{background:#0f1538;border:1px solid #273469;border-radius:6px;padding:2px 6px}</style></head>
<body>
  <div class="box">
    <h1>SimuNet API</h1>
    <p>Status: <strong>{db_state}</strong></p>
    <p>Try: <a href="/status">/status</a> · <a href="/docs">/docs</a></p>
  </div>
</body></html>""")

