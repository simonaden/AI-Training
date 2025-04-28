from fastapi import FastAPI
import uvicorn
import os

app = FastAPI()

#Start the training of the modell
@app.get("/model-training")
async def run_model_training():
    try:
        #Path to the Script
        script_path = os.path.join("scripts", "model_training.py")
        #Trys running the script and getting the output
        command = f"python {script_path}"
        stream = os.popen(command)
        output = stream.read()
        return_code = stream.close()

        if return_code is None:
            return{"output": output, "status": "success"}
        else:
            return{"output": output, "status": "failed", "exit_code": return_code}
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}
    

#Move the model into the wanted folder
@app.get("/export-model")
async def export_model():
    pass

#Merge the new model parameters with the old base model
@app.get("/merge-model")
async def run_merge_model():
    try:
        #Path to the Script
        script_path = os.path.join("scripts", "model_merger.py")
        #Trys running the script and getting the output
        command = f"python {script_path}"
        stream = os.popen(command)
        output = stream.read()
        return_code = stream.close()

        if return_code is None:
            return{"output": output, "status": "success"}
        else:
            return{"output": output, "status": "failed", "exit_code": return_code}
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}

#convert the model folder to gguf format
    #llama.cpp
    #model converter hf to gguf

#create a new model with ollama and a ./Modelfile (Maybe you need to put the gguf in a shared folder with the docker so the docker ollama can run the model and not only the local version)
