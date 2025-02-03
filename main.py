""" main file """
import subprocess
from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.get("/image/describe/{text}")

async def image(text: str, url: Union[str, None] = None):
    """ function to call the smolvlm.py script """
    script_path = "/media/goodsie/codes/Repos/pythin/image/describe/smolvlm.py"
    args = [text]
    if url:
        args.append(url)

    result = subprocess.run(
      ['python', script_path] + args,
      capture_output=True,
      text=True,
      check=True
    )

    return {"output":result.stdout}
