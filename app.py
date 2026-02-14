# This file allows Render to find the app at 'app:app'
# It simply imports the app from main.py

from main import app

# This is needed because Render looks for 'app:app' by default
# If you change the start command to 'uvicorn main:app', you won't need this file

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
