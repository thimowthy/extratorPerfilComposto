import os
from extrairTopoPoco import extrair_constante_topo
from extrairImagens import extractImages
from clusterizarImages import getClustering
from criarCSV import createDepthLabelTable
from agruparImagens import criarPastasClusters


if __name__=="__main__":
    
    baseFolder = os.getcwd()
        
    PDFFolder = os.path.join(baseFolder, "PDF")
    imgFolder = os.path.join(baseFolder, "RECORTES")
    agrupamentoFolder = os.path.join(baseFolder, "AGRUPAMENTO")
    
    clusteringPath = os.path.join(baseFolder, "clustering_final.pkl")
    
    extrair_constante_topo(PDFFolder, imgFolder)
    extractImages(imgFolder)
    clusteringPath = getClustering(imgFolder, clusteringPath)
    createDepthLabelTable(imgFolder)
    criarPastasClusters(clusteringPath, agrupamentoFolder, imgFolder)