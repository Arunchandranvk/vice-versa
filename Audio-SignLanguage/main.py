{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "from fastapi import FastAPI, Request\n",
    "from fastapi.responses import JSONResponse, HTMLResponse\n",
    "from fastapi.staticfiles import StaticFiles\n",
    "from fastapi.templating import Jinja2Templates\n",
    "import your_audio_processing_module  # Import your existing code here\n",
    "\n",
    "app = FastAPI()\n",
    "\n",
    "# Mount static files (for CSS and GIFs)\n",
    "app.mount(\"/static\", StaticFiles(directory=\"static\"), name=\"static\")\n",
    "\n",
    "# Initialize Jinja2 templates\n",
    "templates = Jinja2Templates(directory=\"templates\")\n",
    "\n",
    "@app.get(\"/\", response_class=HTMLResponse)\n",
    "async def index(request: Request):\n",
    "    # Render the HTML page\n",
    "    return templates.TemplateResponse(\"index.html\", {\"request\": request})\n",
    "\n",
    "@app.post(\"/process_audio\", response_class=JSONResponse)\n",
    "async def process_audio():\n",
    "    # Transcribe audio and process text\n",
    "    text = your_audio_processing_module.transcribe_audio()\n",
    "    if text:\n",
    "        processed_text = your_audio_processing_module.process_text(text)\n",
    "        gifs = []\n",
    "        for word in processed_text.split():\n",
    "            gif_path = f\"static/gifs/{word}.gif\"  # Path to GIF files\n",
    "            gifs.append(gif_path)\n",
    "        return {\"gifs\": gifs}\n",
    "    return {\"error\": \"No audio detected\"}\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
