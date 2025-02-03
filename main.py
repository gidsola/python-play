from typing import Union
from fastapi import FastAPI
import subprocess

app = FastAPI();

@app.get("/image/describe/{text}")
async def image(text: str, url: Union[str, None] = None):
  script_path = "/media/goodsie/codes/Repos/pythin/image/describe/SmolVLM.py";
  args = [text];
  if url: args.append(url)
    
  result = subprocess.run(
    ['python', script_path] + args, 
    capture_output=True, 
    text=True
  );
  
  return {"output":result.stdout};
