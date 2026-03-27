import os

import pandas as pd
import torch

# from RNABERT import RNABERT
from torch.nn import AdaptiveAvgPool1d

from scprint import utils

from scprint.tokenizers.protein_embedder import PROTBERT


def protein_embeddings_generator(#用于生成一组基因的蛋白质嵌入表示
    genedf: pd.DataFrame,#包含基因信息的数据框。
    organism: str = "homo_sapiens",  # mus_musculus,
    cache: bool = True,#如果为True，该函数将使用可用的缓存数据。默认为True。
    fasta_path: str = "/tmp/data/fasta/",#fasta文件所在目录的路径。默认为“/tmp/data/fasta/”。
    embedding_size: int = 512,#要生成的嵌入的大小。默认为512。
):
    """
    protein_embeddings_generator embed a set of genes using fasta file and LLMs  protein_embeddings_generator使用fasta文件和 LLM 嵌入一组基因

    Args:
        genedf (pd.DataFrame): A DataFrame containing gene information. 
        organism (str, optional): The organism to which the genes belong. Defaults to "homo_sapiens".
        cache (bool, optional): If True, the function will use cached data if available. Defaults to True.
        fasta_path (str, optional): The path to the directory where the fasta files are stored. Defaults to "/tmp/data/fasta/".
        embedding_size (int, optional): The size of the embeddings to be generated. Defaults to 512.

    Returns:
        pd.DataFrame: Returns a DataFrame containing the protein embeddings, and the RNA embeddings.返回包含蛋白质嵌入和RNA嵌入的DataFrame。
    """
    # given a gene file and organism 给定一个基因文件和生物体
    # load the organism fasta if not already done 如果还没有完成，就快速加载生物体
    utils.load_fasta_species(species=organism, output_path=fasta_path, cache=cache)#Step 1:加载指定物种的 FASTA 文件。如果 cache=True，优先使用已缓存的文件；否则重新下载或生成。
    # subset the fasta 裁剪
    fasta_file = next(#查找物种的全基因组 FASTA 文件（以 .all.fa.gz 结尾）
        file for file in os.listdir(fasta_path) if file.endswith(".all.fa.gz")
    )
    protgenedf = genedf[genedf["biotype"] == "protein_coding"]#Step 2: 筛选蛋白质编码基因：从输入的 genedf 中筛选出 biotype 为 "protein_coding" 的基因子集。
    utils.utils.run_command(["gunzip", fasta_path + fasta_file])#Step 3: 解压并子集化 FASTA 文件：使用 gunzip 解压该文件。
    utils.subset_fasta(#调用 utils.subset_fasta 提取与目标基因对应的蛋白质序列，并保存到 subset.fa 文件中。
        protgenedf.index.tolist(),
        subfasta_path=fasta_path + "subset.fa",
        fasta_path=fasta_path + fasta_file[:-3],
        drop_unknown_seq=True,
    )
    # subset the gene file 
    #Step 4: 生成蛋白质嵌入表示  embed 
    prot_embedder = PROTBERT()#初始化 PROTBERT 实例，调用其 __call__ 方法生成蛋白质序列的嵌入表示。
    prot_embeddings = prot_embedder(
        fasta_path + "subset.fa", output_folder=fasta_path + "esm_out/", cache=cache
    )#结果存储在 fasta_path + "esm_out/" 目录中。
    #Step 5: 清理临时文件  load the data and erase / zip the rest 
    utils.utils.run_command(["gzip", fasta_path + fasta_file[:-3]])#将解压后的 FASTA 文件重新压缩，清理中间文件。
    # return the embedding and gene file
    # TODO: to redebug
    # do the same for RNA
    # rnagenedf = genedf[genedf["biotype"] != "protein_coding"]
    # fasta_file = next(
    #    file for file in os.listdir(fasta_path) if file.endswith(".ncrna.fa.gz")
    # )
    # utils.utils.run_command(["gunzip", fasta_path + fasta_file])
    # utils.subset_fasta(
    #    rnagenedf["ensembl_gene_id"].tolist(),
    #    subfasta_path=fasta_path + "subset.ncrna.fa",
    #    fasta_path=fasta_path + fasta_file[:-3],
    #    drop_unknown_seq=True,
    # )
    # rna_embedder = RNABERT()
    # rna_embeddings = rna_embedder(fasta_path + "subset.ncrna.fa")
    ## Check if the sizes of the cembeddings are not the same
    # utils.utils.run_command(["gzip", fasta_path + fasta_file[:-3]])
    #Step 6: 调整嵌入表示的大小
    m = AdaptiveAvgPool1d(embedding_size)#使用 PyTorch 的 AdaptiveAvgPool1d 将嵌入表示调整为目标维度 embedding_size。
    prot_embeddings = pd.DataFrame(
        data=m(torch.tensor(prot_embeddings.values)), index=prot_embeddings.index
    )#将结果转换为 Pandas DataFrame，保留原始索引。
    # rna_embeddings = pd.DataFrame(
    #    data=m(torch.tensor(rna_embeddings.values)), index=rna_embeddings.index
    # )
    # Concatenate the embeddings
    return prot_embeddings  # pd.concat([prot_embeddings, rna_embeddings]) Step 7: 返回结果 包含蛋白质嵌入表示的 Pandas DataFrame。

#用途：根据输入的基因信息和物种名称，生成蛋白质序列的嵌入表示。 支持缓存机制，避免重复计算。 返回包含蛋白质嵌入表示的 Pandas DataFrame。