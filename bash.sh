```bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi.llvms```
ollama pull llama3.2
echo "FROM llama3.2" >> Modelfile
echo "SYSTEM You are a friendly assistant." >> Modelfile
ollama create -f Modelfile lmlm/Lmkm
ollama push lmlm/Lmkm
