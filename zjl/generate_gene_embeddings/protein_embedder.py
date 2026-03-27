import os

import pandas as pd
from torch import load

from scprint.utils.utils import run_command

# https://github.com/agemagician/ProtTrans
# https://academic.oup.com/nargab/article/4/1/lqac012/6534363


class PROTBERT:#定义了一个名为 PROTBERT 的类，用于调用预训练的蛋白质语言模型（如 ESM-2）来编码蛋白质序列。
    def __init__(#初始化了 PROTBERT 类的实例。
        self,
        config: str = "esm-extract",
        pretrained_model: str = "esm2_t33_650M_UR50D",
    ):
        """
        PROTBERT a ghost class to call protein LLMs to encode protein sequences. PROTBERT是一个伪类，用于调用蛋白质LLM model 来编码蛋白质序列。

        Args:
            config (str, optional): The configuration for the model. Defaults to "esm-extract". 模型的配置。默认为“esm-extract”。
            pretrained_model (str, optional): The pretrained model to be used. Defaults to "esm2_t33_650M_UR50D".要使用的预训练模型。默认为“esm2_t33_650M_UR50D”。
        """
        self.config = config
        self.pretrained_model = pretrained_model

    def __call__(#__call__ 方法使 PROTBERT 实例可以直接像函数一样调用。
        self, input_file: str, output_folder: str = "/tmp/esm_out/", cache: bool = True
    ) -> pd.DataFrame:
        """
        Call the PROTBERT model on the input file. 在输入文件上调用PROTBERT模型。

        Args:
            input_file (str): The input file to be processed.  要处理的输入文件。
            output_folder (str, optional): The folder where the output will be stored. Defaults to "/tmp/esm_out/". 存储输出的文件夹。默认为“/tmp/esm_out/”。
            cache (bool, optional): If True, use cached data if available. Defaults to True. 如果为True，则使用可用的缓存数据。默认为True。

        Returns:
            pd.DataFrame: The results of the model as a DataFrame. 作为DataFrame的模型结果。
        """
        if not os.path.exists(output_folder) or not cache:#检查缓存：如果 output_folder 中已存在结果文件且 cache=True，则跳过计算，直接读取缓存。
            os.makedirs(output_folder, exist_ok=True)#创建输出文件夹并运行命令生成结果。
            print("running protbert")
            cmd = (
                self.config
                + " "
                + self.pretrained_model
                + " "
                + input_file
                + " "
                + output_folder
                + " --include mean"
            )
            try:
                run_command(cmd, shell=True)
            except Exception as e:
                raise RuntimeError(
                    "An error occurred while running the esm-extract command: " + str(e)
                )
        return self.read_results(output_folder)#调用 read_results 方法读取生成的 .pt 文件，并将其转换为 Pandas DataFrame。

    def read_results(self, output_folder):#read_results 方法负责从输出文件夹中读取 .pt 文件，并将它们转换为 Pandas DataFrame。
        """
        Read multiple .pt files in a folder and convert them into a DataFrame. 读取文件夹中的多个.pt文件并将它们转换为DataFrame。

        Args:
            output_folder (str): The folder where the .pt files are stored. 存放.pt文件的文件夹。

        Returns:
            pd.DataFrame: The results of the model as a DataFrame. 作为DataFrame的模型结果。
        """
        files = os.listdir(output_folder)
        files = [i for i in files if i.endswith(".pt")] #获取 output_folder 中的所有 .pt 文件。
        results = [] #创建结果列表
        for file in files:
            results.append(
                load(output_folder + file)["mean_representations"][33].numpy().tolist()#遍历每个 .pt 文件，加载其内容并提取 mean_representations 层（第 33 层）的数据。
            )
        return pd.DataFrame(data=results, index=[file.split(".")[0] for file in files])#转换为 DataFrame

#PROTBERT 是一个“伪类”（ghost class），旨在简化蛋白质序列编码的过程。它通过调用外部工具（如 esm-extract）来生成蛋白质序列的嵌入表示，并将结果存储为 Pandas DataFrame。
#用途：提供一种简单的方式来使用预训练的蛋白质语言模型（如 ESM-2）。支持缓存机制，避免重复计算。将输出结果转换为便于后续分析的格式（Pandas DataFrame）。