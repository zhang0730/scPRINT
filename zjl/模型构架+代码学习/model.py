# from scprint.base.base_model import BaseModel
import copy
import datetime
import os
from functools import partial

# from galore_torch import GaLoreAdamW
from math import factorial
from pathlib import Path
from typing import Dict, Optional

import lightning as L
import pandas as pd
import torch
import torch.distributed
from huggingface_hub import PyTorchModelHubMixin
from lightning.pytorch.callbacks.lr_finder import LearningRateFinder
from lightning.pytorch.tuner.lr_finder import _LRCallback
from scipy.sparse import load_npz
from simpler_flash import FlashTransformer
from torch import Tensor, nn, optim

# from .linear_transformer import FastTransformerEncoderWrapper as FastTransformer
from . import decoders, encoders, fsq, loss, utils
from .loss import grad_reverse
from .utils import WeightedMasker, simple_masker

FILEDIR = os.path.dirname(os.path.realpath(__file__))


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


class scPrint(L.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        genes: list,#模型将使用的基因名称列表。
        organisms: list = ["NCBITaxon:9606"],
        precpt_gene_emb: Optional[str] = None,#np.数组，可选： 大小为len(genes), d_model的基因嵌入。顺序应与基因相同。默认为 “无”。
        gene_pos_enc: Optional[list] = None,#列表，可选： 与基因大小相同的基因位置编码。为基因中的每个基因提供一个位置值。默认为 “无”。
        normalization: str = "sum",
        d_model: int = 512,#int，可选项： 模型的维度。默认为 512。
        nhead: int = 8,#int，可选项： 多头注意力模型中的头数。默认为 8。同scGPT
        attn_bias: str = "none",
        d_hid: int = 512,#int，可选项： 前馈网络模型的维度。默认为 512。在scGPT是MLP 多层感知器
        edge_dim: int = 12,
        nlayers: int = 6, #int，可选项：transformer模型的层数。默认为 6。在scGPT是12层
        expr_encoder_layers: int = 2,#int，可选：表达量 编码器 的层数：默认为 2。
        layers_cls: list[int] = [],#list[int]，可选项： 指定分类器层数的列表。默认为[]。
        classes: Dict[str, int] = {},#Dict[str, int]，可选： 要预测的类别以及每个类别的数量。默认为 {}。
        labels_hierarchy: Dict[str, Dict[int, list[int]]] = {}, #Dict[str, Dict[int, list[int]]]，可选）： 具有分层类别的类别层次结构。默认为 {}。
        dropout: float = 0.1,
        transformer: str = "fast",#str，可选项： 要使用的变换器类型。One of "linear", "flash", "flashsparse", "scprint". Defaults to "fast".
        expr_emb_style: str = "continuous",  #str，可选项：输入嵌入的样式： 输入嵌入风格。One of "continuous", "binned_pos", "cont_pos". Defaults to "continuous".
        domain_spec_batchnorm: str = "None",#str，可选项： 是否应用特定域的批量规范化。默认为 “无”。
        n_input_bins: int = 0,
        num_batch_labels: int = 0,
        mvc_decoder: str = "None",#str，可选项）： MVC 解码器的样式：One of "None", "inner product", "concat query", "sum query". Defaults to "None".
        pred_embedding: list[str] = [],#list[str]，可选项： 用于绘制嵌入的类列表。默认为[]。
        cell_emb_style: str = "cls",#str，可选项：细胞嵌入的样式：One of "cls", "avg-pool", "w-pool". Defaults to "cls".
        cell_specific_blocks: bool = False,
        depth_atinput: bool = True,
        freeze_embeddings: bool = True,#bool，可选项： 是否在训练过程中冻结嵌入。默认为 True。
        label_decoders: Optional[Dict[str, Dict[int, str]]] = None,#可选[Dict[str, Dict[int, str]]] 在验证过程中绘制 UMAP 时使用的标签解码器。默认为 “无”。
        zinb: bool = True,#bool，可选： 是否使用零膨胀负二项分布。默认为 True。用于expression decoder
        lr: float = 0.0001,
        compress_class_dim: Optional[Dict[str, int]] = None,
        **flash_attention_kwargs,#dict：模型的附加关键字参数： 参见 @flashformer.py
    ):
        """
        scPRINT transformer for single cell biology and the inference of Gene Regulatory networks

        Args:参数
            genes (list): List of gene names the model will work with.
            precpt_gene_emb (np.array, optional): Gene embeddings of size (len(genes), d_model). Should be in the same order as the genes. Defaults to None.
            gene_pos_enc (list, optional): Gene position encoding of the same size as genes. Provides a location value for each gene in genes. Defaults to None.
            d_model (int, optional): Dimension of the model. Defaults to 512.
            nhead (int, optional): Number of heads in the multihead attention models. Defaults to 8.
            d_hid (int, optional): Dimension of the feedforward network model. Defaults to 512.
            nlayers (int, optional): Number of layers in the transformer model. Defaults to 6.
            expr_encoder_layers (int, optional): Number of layers in the expression encoder. Defaults to 2.
            layers_cls (list[int], optional): List specifying the number of layers in the classifier. Defaults to [].
            classes (Dict[str, int], optional): Classes to predict with the number of classes for each. Defaults to {}.
            labels_hierarchy (Dict[str, Dict[int, list[int]]], optional): Class hierarchy for classes with hierarchical classes. Defaults to {}.
            dropout (float, optional): Dropout value. Defaults to 0.2.
            transformer (str, optional): Transformer type to use. One of "linear", "flash", "flashsparse", "scprint". Defaults to "fast".
            domain_spec_batchnorm (str, optional): Whether to apply domain-specific batch normalization. Defaults to "None".
            expr_emb_style (str, optional): Style of input embedding. One of "continuous", "binned_pos", "cont_pos". Defaults to "continuous".
            mvc_decoder (str, optional): Style of MVC decoder. One of "None", "inner product", "concat query", "sum query". Defaults to "None".
            pred_embedding (list[str], optional): List of classes to use for plotting embeddings. Defaults to [].
            cell_emb_style (str, optional): Style of cell embedding. One of "cls", "avg-pool", "w-pool". Defaults to "cls".
            freeze_embeddings (bool, optional): Whether to freeze the embeddings during training. Defaults to True.
            label_decoders (Optional[Dict[str, Dict[int, str]]], optional): Label decoders to use for plotting the UMAP during validations. Defaults to None.
            zinb (bool, optional): Whether to use Zero-Inflated Negative Binomial distribution. Defaults to True.
            **flash_attention_kwargs (dict): Additional keyword arguments for the model. see @flashformer.py

        Notes:
            for other parameters of the model that are not part of its class definition, see @trainer.trainer.py

        Raises:
            ValueError: If the expr_emb_style is not one of "continuous", "binned_pos", "cont_pos".
        """
        super().__init__() #确保父类的初始化代码被执行，会有父类的属性
        self.save_hyperparameters()#自动将所有以self为前缀的变量保存为模型的超参数。
        # training flags
        self.do_denoise = True #控制是否进行去噪以及噪声水平。
        self.noise = [0.6] #控制噪声水平。
        self.do_cce = False#是否启用某种与交叉熵（cross-entropy）相关的操作。
        self.cce_temp = 0.2#交叉熵的温度参数（temperature parameter），设置为 0.2。
        self.lr = 0.0001 #学习率，控制优化器更新权重的速度。
        self.cce_scale = 0.1#交叉熵的缩放因子，设置为 0.1。
        self.do_ecs = False
        self.ecs_threshold = 0.4#ECS 的阈值，设置为 0.4。
        self.ecs_scale = 0.1#ECS 的缩放因子，设置为 0.1。
        self.do_mvc = False#是否进行多视图融合（multiview consenseus)
        self.mvc_scale = 1.0#MVC 的缩放因子，设置为 1.0。
        self.class_embd_diss_scale = 0.1#类嵌入（class embedding）的不相似性缩放因子，设置为 0.1。
        self.do_adv_cls = False #是否启用对抗性分类（adversarial classification）。
        self.adv_class_scale = 0.1#对抗性分类的缩放因子，设置为 0.1。
        self.do_cls = False#是否启用分类任务。
        self.mean_attn_tot = None#用于存储注意力机制的平均值。
        self.mean_attn_tot_c = 0#与注意力机制相关的计数器。
        self.do_adv_batch = False#是否启用对抗性批量操作。
        self.run_full_forward = True#是否运行完整的前向传播。
        self.class_scale = 1#分类任务的缩放因子，设置为 1。
        self.zinb_and_mse = False#是否使用零膨胀负二项分布（ZINB）和均方误差（MSE）。
        self.do_next_tp = False#是否启用某种与“下一步”（next step）相关的操作。
        self.do_generate = False#是否启用生成任务。
        self.var_context_length = False#是否使用可变上下文长度。
        self.depth_atinput = depth_atinput#输入深度（depth_atinput）的值，具体含义取决于上下文。
        self.mask_ratio = []# 掩码比例（mask ratio），用于控制数据中被掩码的部分。
        self.warmup_duration = 500#学习率预热（warm-up）的持续时间，设置为 500。
        self.weight_decay = 0.01#权重衰减（L2 正则化）的系数，设置为 0.01。
        self.optim = "adamW"# 优化器类型，设置为 "adamW"。
        self.fused_adam = False# 是否使用融合的 Adam 优化器。
        self.lr_reduce_patience = 2#学习率衰减的耐心（patience），设置为 2。
        self.lr_reduce_factor = 0.6#学习率衰减因子，设置为 0.6。
        self.test_every = 20# 每隔多少次迭代进行一次测试，设置为 20。
        self.lr_reduce_monitor = "val_loss"#监控的指标，用于决定是否衰减学习率，设置为 "val_loss"。
        self.name = ""
        self.lrfinder_steps = 0#学习率查找器（lr finder）的步数，设置为 0。
        self.doplot = True
        self.get_attention_layer = []#用于获取注意力层的列表。
        self.embs = None#嵌入（embedding）的值。
        self.pred_log_adata = True
        self.attn = utils.Attention(#初始化了一个注意力机制（attention mechanism）的实例。
            len(genes),
            additional_tokens=(
                len(classes) + (2 if self.depth_atinput else 1)
                if not cell_specific_blocks
                else 0
            ),
        )
        self.tf_masker = WeightedMasker(genes, inv_weight=0.05)#初始化一个 WeightedMasker 模块，用于对基因数据进行加权掩码操作。inv_weight=0.05 表示掩码操作的权重。
        self.predict_depth_mult = 3#预测深度的倍数，设置为 3。
        self.predict_mode = "none"#预测模式，设置为 "none"。
        self.keep_all_cls_pred = False#是否保留所有分类预测结果，设置为 False。
        self.cell_separation = True# 是否启用细胞分离操作，设置为 True。
        # should be stored somehow 就是把初始化时的参数全部储存起来
        self.d_model = d_model#模型的维度（例如 Transformer 中的隐藏层维度）。
        self.normalization = normalization#归一化方法（例如 LayerNorm 或 BatchNorm）。
        self.organisms = organisms#涉及的生物种类。
        self.edge_dim = edge_dim#边的维度（可能用于图神经网络）。上面的默认是12
        self.attn_bias = attn_bias#注意力机制的偏置。
        self.nlayers = nlayers#模型的层数。
        self.gene_pos_enc = gene_pos_enc#与基因大小相同的基因位置编码。
        self.mvc_decoder = mvc_decoder# MVC 解码器
        self.domain_spec_batchnorm = domain_spec_batchnorm#str，可选项： 是否应用特定域的批量规范化。默认为 “无”。
        # need to store
        self.n_input_bins = n_input_bins#输入数据的分箱数量。难道这也跟scGPT相同吗？
        self.transformer = transformer#str，可选项： 要使用的变换器类型。One of "linear", "flash", "flashsparse", "scprint". Defaults to "fast".
        self.label_counts = classes # 要预测的类别以及每个类别的数量。默认为 {}。字典形状
        self.classes = list(classes.keys())#Classes to predict with the number of classes for each. Defaults to {}.

        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:#细胞嵌入风格
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")#可以是 "cls"（使用分类标记）、"avg-pool"（平均池化）或 "w-pool"（加权池化）。
        self.cell_emb_style = cell_emb_style

        self.label_decoders = label_decoders#Label decoders to use for plotting the UMAP during validations.
        self.pred_embedding = pred_embedding#List of classes to use for plotting embeddings.
        self.genes = genes#List of gene names the model will work with.
        self.vocab = {i: n for i, n in enumerate(genes)}#基因的词汇表，将基因索引映射到基因名称。
        self.expr_emb_style = expr_emb_style#Style of input embedding. One of "continuous", "binned_pos", "cont_pos". Defaults to "continuous".
        if self.expr_emb_style not in ["category", "continuous", "none"]:
            raise ValueError(
                f"expr_emb_style should be one of category, continuous, scaling, "
                f"got {expr_emb_style}"
            )
        self.labels_hierarchy = labels_hierarchy#Class hierarchy for classes with hierarchical classes. Defaults to {}.
        #超参数的存储 将一些重要的配置存储到 self.hparams 中，方便后续的记录和复现。
        self.hparams["labels_hierarchy"] = self.labels_hierarchy
        self.hparams["classes"] = self.classes
        self.hparams["label_decoders"] = self.label_decoders
        self.hparams["label_counts"] = self.label_counts
        self.hparams["gene_pos_enc"] = self.gene_pos_enc
        self.hparams["genes"] = self.genes

        self.mat_labels_hierarchy = {}#标签层次结构的矩阵化
        for k, v in labels_hierarchy.items():#labels_hierarchy 的每个键值对 (k, v)
            tens = torch.zeros((len(v), classes[k]))#对于每个子字典 v，创建一个零矩阵 tens
            for k2, v2 in v.items():#遍历子字典 v 的键值对 (k2, v2)，将tens矩阵中对应的值设置为 1
                tens[k2 - classes[k], v2] = 1
            self.mat_labels_hierarchy[k] = tens.to(bool)#将矩阵转换为布尔类型，并存储到 self.mat_labels_hierarchy 中

        # encoder 编码器架构，用于处理基因表达数据和其他相关信息。 这个比scGPT要复杂一点
        # gene encoder 初始化基因编码器，用于将基因嵌入到模型的隐藏空间中。
        if precpt_gene_emb is not None:#如果提供了预训练的基因嵌入文件（precpt_gene_emb），则从文件中加载基因嵌入。
            embeddings = pd.read_parquet(precpt_gene_emb).loc[self.genes]#使用 pandas.read_parquet 读取嵌入文件，并通过 .loc[self.genes] 筛选出模型需要的基因。
            if len(embeddings) == 0:#如果嵌入文件中没有包含任何模型需要的基因，抛出错误。
                raise ValueError(
                    f"the gene embeddings file {precpt_gene_emb} does not contain any of the genes given to the model"
                )
            elif len(embeddings) < len(self.genes):#如果嵌入文件中只包含部分基因，打印警告信息。
                print(
                    "Warning: only a subset of the genes available in the embeddings file."
                )
                print("number of genes: ", len(embeddings))
            sembeddings = torch.nn.AdaptiveAvgPool1d(d_model)(#使用 torch.nn.AdaptiveAvgPool1d 将嵌入的维度调整为目标维度 d_model。
                torch.tensor(embeddings.values)
            )

            self.gene_encoder = encoders.GeneEncoder(#初始化 GeneEncoder，将离散的输入（在这里是基因或词汇表中的单词）映射到连续的向量空间，即生成嵌入表示；并传入预训练权重sembeddings。如果设置了 freeze_embeddings，则冻结这些嵌入。
                len(self.vocab), d_model, weights=sembeddings, freeze=freeze_embeddings
            )
        else:
            self.gene_encoder = encoders.GeneEncoder(len(self.vocab), d_model)#如果没有提供预训练嵌入文件，则初始化一个默认的 GeneEncoder，不使用预训练权重。

        # Value Encoder, NOTE: the scaling style is also handled in _encode method  表达值编码器（Expression Value Encoder）  初始化表达值编码器，用于处理基因表达值。 根据 expr_emb_style 的值选择编码器类型：
        if expr_emb_style in ["continuous", "full_pos"]:# 如果是 "continuous" 或 "full_pos"，使用 ContinuousValueEncoder，适用于连续表达值。
            self.expr_encoder = encoders.ContinuousValueEncoder(
                d_model, dropout, layers=expr_encoder_layers
            )
        elif expr_emb_style == "binned_pos":#如果是 "binned_pos"，使用 CategoryValueEncoder，适用于分箱后的表达值。
            assert n_input_bins > 0
            self.expr_encoder = encoders.CategoryValueEncoder(n_input_bins, d_model)
        else:
            self.expr_encoder = torch.nn.Identity()#如果是其他类型（如 "none"），则不进行编码，直接使用 torch.nn.Identity。

        # Positional Encoding 位置编码器（Positional Encoding） 初始化位置编码器，用于为基因添加位置信息。
        if self.gene_pos_enc is not None:#如果 self.gene_pos_enc 不为 None，则计算最大长度 max_len。
            max_len = max(gene_pos_enc)
            token_to_pos = {token: pos for token, pos in enumerate(self.gene_pos_enc)}#创建一个映射表 token_to_pos，将基因索引映射到位置。
            self.pos_encoder = encoders.PositionalEncoding(#初始化 PositionalEncoding，将位置信息嵌入到模型的隐藏空间中。
                d_model, max_len=max_len, token_to_pos=token_to_pos
            )

        self.cell_embs_count = len(self.classes) + (2 if self.depth_atinput else 1)#初始化细胞嵌入 计算细胞嵌入的数量 self.cell_embs_count，包括类别嵌入和其他可能的嵌入（如时间嵌入和深度嵌入）。classses是一开始的时候定义的类别
        # Class Encoder
        # always have [base_cell_emb, time_embedding, depth_embedding] + any other class info
        # base cell embedding will store other cell specific information
        self.class_encoder = encoders.CategoryValueEncoder(#初始化 CategoryValueEncoder 作为类别编码器。
            self.cell_embs_count - (1 if self.depth_atinput else 0), d_model
        )
        # self.time_encoder = encoders.ContinuousValueEncoder(d_model, dropout)
        self.depth_encoder = encoders.ContinuousValueEncoder(#初始化 ContinuousValueEncoder 作为深度编码器。这里也有深度的考虑吗 不同于scfoundation的RDA模型
            d_model, dropout, layers=expr_encoder_layers
        )

        # compute tensor for mat_labels_hierarchy 标签层次结构的张量计算 从 flash_attention_kwargs 中移除一些不相关的参数，以避免在初始化 Transformer 时引发错误。
        for i in ["strict_loading", "optim", "weight_decay", "lr"]:
            if i in flash_attention_kwargs:
                flash_attention_kwargs.pop(i)
        # Transformer 初始化 Transformer 模型。
        # Linear
        if transformer == "linear":
            # linear transformer using the fast transformer package
            # self.transformer = FastTransformerEncoder(
            #    d_model, nhead, d_hid, nlayers, dropout, "linear"
            # )
            raise NotImplementedError("Linear transformer is not implemented")#如果指定的 Transformer 类型是 "linear"，则抛出未实现错误。
        # regular or flash 如果是 "flash" 或其他类型，使用 FlashTransformer。
        else:
            self.transformer = FlashTransformer(#这是比较精简的transformer启动了吧
                d_model=d_model,#模型的维度
                nhead=nhead,#头的数目
                dropout=dropout,#丢弃的概率
                nlayers=nlayers,#模型的层数
                cross_attn=cell_specific_blocks,#？
                use_flash_attn=(transformer == "flash"),
                **flash_attention_kwargs,#直接启用之前的超参数
            )
        if cell_specific_blocks:#如果启用了 cell_specific_blocks，则额外初始化一个 cell_transformer，用于处理细胞特定的块。
            self.cell_transformer = FlashTransformer(
                d_model=d_model,
                nhead=nhead,
                nlayers=6,
                dropout=dropout,
                cross_attn=True,
                use_flash_attn=(transformer == "flash"),
                **flash_attention_kwargs,
            )
        else:
            self.cell_transformer = None

        # decoders
        # expression 基因表达解码器（Expression Decoder）初始化基因表达解码器，用于从编码后的特征中重建基因表达值。
        self.expr_decoder = decoders.ExprDecoder(
            d_model,
            nfirst_tokens_to_skip=self.cell_embs_count,#跳过的起始标记数量（例如单元格嵌入、时间嵌入等）。
            dropout=dropout,
            zinb=zinb,#是否使用零膨胀负二项分布（Zero-Inflated Negative Binomial, ZINB）来建模基因表达值。 这个是作者比较在意他自己创新的地方
            use_depth=not self.depth_atinput,#是否使用深度信息
        )
        # cls decoder 分类解码器（Classification Decoders）初始化分类解码器，用于从编码后的特征中预测类别标签。
        self.cls_decoders = torch.nn.ModuleDict()#使用 torch.nn.ModuleDict 存储多个分类解码器，每个解码器对应一个类别。
        # should be a very simple classifier for most things
        # (maybe scale with the number of classes) should be 1 layer...
        for clss, n_cls in classes.items():
            self.cls_decoders[clss] = decoders.ClsDecoder(
                d_model, n_cls, layers=layers_cls, dropout=dropout
            )#遍历 classes 字典，为每个类别初始化一个 ClsDecoder。

        # Batch effect correction via adversarial training on batch classes  批量效应校正（Batch Effect Correction）
        if num_batch_labels > 0:#如果存在批量标签（num_batch_labels > 0），则初始化一个对抗性判别器。
            self.grad_reverse_discriminator_loss = loss.AdversarialDiscriminatorLoss(#AdversarialDiscriminatorLoss 用于训练一个判别器，以区分不同批次的数据，从而实现批量效应校正。
                d_model,
                n_cls=num_batch_labels,
            )
        else:
            self.grad_reverse_discriminator_loss = None#如果没有批量标签，则将判别器设置为 None。

        # expression decoder from batch embbedding MVC 解码器（MVC Decoder）
        if mvc_decoder != "None":
            self.mvc_decoder = decoders.MVCDecoder(#初始化多视图一致性（MVC）解码器。
                d_model,
                arch_style=mvc_decoder,
                zinb=zinb,
            )
        else:
            self.mvc_decoder = None

        #初始化权重 对模型的所有模块应用权重初始化函数。
        self.apply(#使用 self.apply 遍历模型的所有子模块。
            partial(
                utils._init_weights,#应用 utils._init_weights 函数进行权重初始化，其中 n_layer 是模型的层数。
                n_layer=nlayers,
            )
        )
        #分类解码器的输出层偏置初始化
        for i, dec in self.cls_decoders.items():#遍历 self.cls_decoders 中的每个解码器。
            torch.nn.init.constant_(dec.out_layer.bias, -0.13)#使用 torch.nn.init.constant_ 将输出层的偏置初始化为 -0.13。

        if compress_class_dim is not None:#类别维度压缩（Bottleneck MLPs）
            self.bottleneck_mlps = torch.nn.ModuleDict()#如果 compress_class_dim 不为 None，则初始化一个 torch.nn.ModuleDict 存储压缩模块。
            for k, v in compress_class_dim.items():
                self.bottleneck_mlps[k] = fsq.FSQ(levels=[2] * v, dim=self.d_model)#遍历 compress_class_dim，为每个类别初始化一个 fsq.FSQ 模块。
        else:
            self.bottleneck_mlps = None

    def on_load_checkpoint(self, checkpoints):#加载模型检查点（checkpoint）时被调用。它的主要作用是处理检查点加载过程中可能出现的不一致情况，例如模型结构的变化、类别数量的变化、超参数的更新等。
        for name, clss in self.cls_decoders.items():# 动态调整分类解码器的输出层，以匹配检查点中的类别数量。 遍历 self.cls_decoders 中的每个分类解码器。
            size = checkpoints["state_dict"][
                "cls_decoders." + name + ".out_layer.bias"
            ].shape[0]#从检查点的 state_dict 中获取对应解码器的输出层偏置的大小。
            if size != clss.out_layer.bias.shape[0]:#如果检查点中的大小与当前模型的大小不一致，则重新初始化该解码器的输出层，使其匹配检查点中的大小。
                self.cls_decoders[name].out_layer = torch.nn.Linear(
                    clss.out_layer.weight.shape[1], size
                )
        size = checkpoints["state_dict"]["class_encoder.embedding.weight"].shape[0]
        if size != self.class_encoder.embedding.weight.shape[0]:#动态调整类别编码器的嵌入大小。
            self.class_encoder = encoders.CategoryValueEncoder(size, self.d_model)#如果检查点中的大小与当前模型的大小不一致，则重新初始化类别编码器，并更新 self.cell_embs_count。
            self.cell_embs_count = size
            print("changing size, could lead to issues")#打印警告信息，提示大小变化可能导致问题。
        size = checkpoints["state_dict"][#动态调整对抗性判别器的输出层，以匹配检查点中的类别数量。
            "grad_reverse_discriminator_loss.out_layer.bias"
        ].shape[0]#从检查点的 state_dict 中获取判别器输出层偏置的大小。
        # we won't use it but still need to take care of it. for now will still add it to the model
        if size != self.grad_reverse_discriminator_loss.out_layer.bias.shape[0]:#如果检查点中的大小与当前模型的大小不一致，则重新初始化判别器。
            self.grad_reverse_discriminator_loss = loss.AdversarialDiscriminatorLoss(
                self.d_model,
                n_cls=size,
            )
            print(
                "the discriminator for batch effect correction has been resized\
                and re-initiliazed. It will start from scratch during this training if "
            )#打印警告信息，提示判别器已被重新初始化，将在本次训练中从头开始。

        # if len(checkpoints["state_dict"]["pos_encoder.pe"].shape) == 3:
        #    self.pos_encoder.pe = checkpoints["state_dict"]["pos_encoder.pe"].squeeze(1)

        #更新超参数和类别信息 从检查点中加载超参数和类别信息，并更新模型的状态。
        self.normalization = checkpoints["hyper_parameters"]["normalization"]#更新归一化方法 self.normalization。
        if "classes" in checkpoints["hyper_parameters"]:#如果检查点中包含类别信息，更新类别数量、标签计数、标签解码器和标签层次结构。
            if self.classes != checkpoints["hyper_parameters"]["classes"]:#类别信息
                print("changing the number of classes, could lead to issues")

            if "label_counts" in checkpoints["hyper_parameters"]:#类别数量
                self.label_counts = checkpoints["hyper_parameters"]["label_counts"]
                self.classes = checkpoints["hyper_parameters"]["classes"]#标签计数
            else:
                self.label_counts = checkpoints["hyper_parameters"]["classes"]
                self.classes = list(self.label_counts.keys())
            self.label_decoders = checkpoints["hyper_parameters"]["label_decoders"]#标签解码器
            self.labels_hierarchy = checkpoints["hyper_parameters"]["labels_hierarchy"]#标签层次结构
            for k, v in self.labels_hierarchy.items():#再一次标签层次结构的矩阵化：更新 self.mat_labels_hierarchy，将标签层次结构转换为布尔矩阵。
                tens = torch.zeros((len(v), self.label_counts[k]))
                for k2, v2 in v.items():
                    tens[k2 - self.label_counts[k], v2] = 1
                self.mat_labels_hierarchy[k] = tens.to(bool)
        if "gene_pos_enc" in checkpoints["hyper_parameters"]:#检查基因信息和位置编码
            if self.genes != checkpoints["hyper_parameters"]["genes"]:#比较当前模型的基因列表和检查点中的基因列表。
                raise ValueError(
                    "Genes or their ordering have changed in the dataloader compared to last time, the model will likely misbehave!"
                )
            if self.gene_pos_enc != checkpoints["hyper_parameters"]["gene_pos_enc"]:#位置编码
                print(
                    "Gene position encoding has changed in the dataloader compared to last time, be careful!"
                )#如果当前模型的类别信息与检查点中的不一致，打印警告信息。
        mencoders = {}#更新数据模块的解码器
        try:
            if self.trainer.datamodule.decoders != self.label_decoders:#如果数据模块中的解码器与检查点中的解码器不一致，更新数据模块的解码器。
                # if we don't have the same decoders, we need to update the one on the datamodule side
                for k, v in checkpoints["hyper_parameters"]["label_decoders"].items():
                    mencoders[k] = {va: ke for ke, va in v.items()}
                self.trainer.datamodule.dataset.mapped_dataset.encoders = mencoders
                if (
                    self.trainer.datamodule.kwargs["collate_fn"].organism_name#如果数据模块的 collate_fn 中包含解码器信息，重新初始化 collate_fn
                    in mencoders
                ):
                    self.trainer.datamodule.kwargs["collate_fn"]._setup(
                        org_to_id=mencoders[
                            self.trainer.datamodule.kwargs["collate_fn"].organism_name
                        ],
                        valid_genes=self.genes,
                    )
            os.environ["MY_SLURM_RESTART_COUNT"] = str(#更新 SLURM 重启计数环境变量。
                int(os.getenv("SLURM_RESTART_COUNT", 0))
                + 1
                + int(os.getenv("MY_SLURM_RESTART_COUNT", 0))
            )
        except RuntimeError as e:#如果捕获到 RuntimeError，打印错误信息。
            if "scPrint is not attached to a `Trainer`." in str(e):
                print("RuntimeError caught: scPrint is not attached to a `Trainer`.")
        if not is_interactive():
            self.save_hyperparameters()

    def _encoder(#将输入数据编码为嵌入向量 embeddings
        self,
        gene_pos: Tensor,
        expression: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        req_depth: Optional[Tensor] = None,
        timepoint: Optional[Tensor] = None,
        cell_embs: Optional[Tensor] = None,  # (minibatch, n_labels, embsize)
    ):
        """
        _encode given inputs to the model encode into embeddings.

        Args:
            @see self.forward()它是一个内部方法，通常在模型的 forward 方法中被调用。

        Returns:
            Tensor: the encoded data 返回一个张量 是被编码的数据
        """
        enc = self.gene_encoder(gene_pos)  # 这里用的是上面模型初始化时初始化的GeneEncoder，使用基因位置编码张量生成嵌入(minibatch, seq_len, embsize) enc是这个函数产生的关键变量
        self.cur_gene_token_embs = enc.clone() #将编码结果存储到 self.cur_gene_token_embs 中，用于后续可能的访问。

        #表达值编码
        if expression is not None:#根据 self.normalization 的值选择归一化方法：
            if self.normalization == "sum":
                norm_expr = expression / expression.sum(1).unsqueeze(1)

            elif self.normalization == "log":
                norm_expr = torch.log2(1 + expression)
            else:
                raise ValueError(f"Unknown normalization: {self.normalization}")
            enc.add_(self.expr_encoder(norm_expr, mask))#使用 self.expr_encoder 对归一化后的表达值进行编码，并将结果加到 enc 中。
        #基因位置编码
        if self.gene_pos_enc:
            enc.add_(self.pos_encoder(gene_pos))#如果启用了基因位置编码，将位置信息编码为嵌入向量并加到 enc 中。这里的pos_encoder位置编码器和表达量编码器一样是在初始化时就定义过的了
        if cell_embs is None:#如果未提供细胞嵌入（cell_embs），则使用类别编码器生成默认的细胞嵌入。
            cell_embs = self.class_encoder(#3.调用 self.class_encoder 对该张量进行编码，生成单元格嵌入。
                torch.arange(
                    self.cell_embs_count - (1 if self.depth_atinput else 0),#1.使用 torch.arange 生成一个序列，长度为 self.cell_embs_count（减去 1 如果启用了深度输入）。
                    device=expression.device,
                ).repeat(expression.shape[0], 1)#2.将该序列重复 minibatch 次，形成形状为 (minibatch, n_labels) 的张量。
            )
        if timepoint is not None:#时间点
            pass
            # cell_embs[:, 2, :] = self.time_encoder(timepoint)
        if req_depth is not None:#如果提供了深度信息（req_depth）：
            # cell_embs = cell_embs.clone()
            # cell_embs[:, 1, :] += self.depth_encoder(torch.log2(1 + req_depth))
            depth_encoded = self.depth_encoder(torch.log2(1 + req_depth)).unsqueeze(1)#先对深度信息进行对数变换（log2(1 + req_depth)），然后使用 self.depth_encoder（上面定义）对其进行编码。
            cell_embs = torch.cat(
                (cell_embs[:, :1, :], depth_encoded, cell_embs[:, 1:, :]), dim=1#将编码后的深度信息插入到细胞嵌入的指定位置。
            )
        return torch.cat([cell_embs, enc], dim=1)  # self.norm_and_dropout(enc)#将细胞嵌入和基因嵌入拼接在一起，返回最终的编码结果。返回形状为 (minibatch, seq_len + n_labels, embsize) 的张量。
        # we already apply prenorm & dropout  # (minibatch, seq_len, embsize)  与scGPT比较 他的输入token包括基因表达值（bin)+基因名（tokenization)+状态标签

    #是模型解码器的核心部分，用于将 Transformer 的输出解码为最终的模型输出。
    def _decoder(
        self,
        transformer_output,
        depth_mult,#深度缩放因子，用于调整输出的规模。
        get_gene_emb=False,#是否返回基因嵌入，默认为 False。
        do_sample=False,#是否进行采样操作，默认为 False。
        do_mvc=False,#是否执行多视图一致性（MVC）任务，默认为 False。
        do_class=False,#是否执行分类任务，默认为 False。
        req_depth: Optional[Tensor] = None,#请求的深度信息，形状为 (minibatch)，可选。
    ):
        """
        _decoder given the transformer output, decode into the final output.

        Args:
            @see self.forward() 它是一个内部方法，通常在模型的 forward 方法中被调用。

        Returns:
            dict: the output of the model 返回一个字典作为模型的输出
        """
        if req_depth is not None:#深度信息的对数变换
            req_depth = torch.log2(1 + req_depth)#对数变换可以稳定深度信息的分布，使其更适合模型处理。
        output = self.expr_decoder(transformer_output, req_depth)#基因表达解码 

        output["mean"] = depth_mult.unsqueeze(1) * output["mean"]#使用深度缩放因子（depth_mult）调整基因表达的规模
        if do_sample:
            pass

        output["cell_embs"] = self.get_cell_embs(transformer_output)#调用 self.get_cell_embs 方法，提取与细胞相关的嵌入信息。
        if self.bottleneck_mlps is not None:#如果启用了类别维度压缩（self.bottleneck_mlps），对每个类别的细胞嵌入进行压缩。
            for i, clsname in enumerate(self.classes):#遍历所有类别（self.classes）。
                loc = i + (2 if self.depth_atinput else 1)
                output["cell_embs"][:, loc, :] = self.bottleneck_mlps[clsname](#对每个类别的细胞嵌入调用对应的 bottleneck_mlps 模块进行压缩。
                    output["cell_embs"][:, loc, :]
                )[0]
        output["cell_emb"] = torch.mean(output["cell_embs"].clone(), dim=1)#计算所有细胞嵌入的平均值，作为细胞的整体嵌入表示。
        if len(self.classes) > 0 and do_class:#如果启用了分类任务（do_class=True），对每个类别调用分类解码器。
            output.update(
                {
                    "cls_output_" + clsname: self.cls_decoders[clsname](#将分类结果存储到 output 字典中，键为 "cls_output_" + clsname。
                        output["cell_embs"][
                            :, i + (2 if self.depth_atinput else 1), :#对每个类别的单元格嵌入调用对应的分类解码器（self.cls_decoders[clsname]）。
                        ]  # the first elem is the base cell embedding
                    )
                    for i, clsname in enumerate(self.classes)#遍历所有类别（self.classes）。
                }
            )  # (minibatch, n_cls)
        if do_mvc:#如果启用了多视图一致性学习（do_mvc=True），调用 MVC 解码器。
            output.update(#更新 output 字典，包含 MVC 解码的结果。
                self.mvc_decoder(output["cell_emb"], self.cur_gene_token_embs)#调用 self.mvc_decoder，输入单元格嵌入（output["cell_emb"]）和当前基因嵌入（self.cur_gene_token_embs）在基因编码的时候clone的。
            )
            output["mvc_mean"] = (
                depth_mult.unsqueeze(1) * output["mvc_mean"]#使用深度缩放因子调整 MVC 的均值输出（output["mvc_mean"]）。
            )  # (minibatch, seq_len)

        if get_gene_emb:#如果需要返回基因嵌入（get_gene_emb=True），从 Transformer 的输出中提取基因嵌入部分。
            output["gene_embedding"] = transformer_output[
                :, self.cell_embs_count :, :
            ]  # (minibatch, seq_len, embsize)#提取 transformer_output 中从 self.cell_embs_count 开始的部分，作为基因嵌入。  张量的形状为(小批量的大小, 序列的长度, 嵌入维度)
        return output

    def forward(#前向传播 模型的核心：将输入数据通过编码器、Transformer 和解码器，生成最终的输出。
        self,
        gene_pos: Tensor,#基因位置信息，形状为 (minibatch, seq_len)。作为gene_encoder的输入
        expression: Optional[Tensor] = None,#基因表达值，形状为 (minibatch, seq_len)，可选。作为gene_encoder的输入
        mask: Optional[Tensor] = None,#掩码信息，用于在前向传递过程中屏蔽序列中的某些元素，形状为 (minibatch, seq_len)，可选。
        req_depth: Optional[Tensor] = None,#每个序列的完整深度。形状为 (minibatch)，可选。
        timepoint: Optional[Tensor] = None,  #每个序列相关的时间点。(new_minibatch_of_nxt_cells,)
        get_gene_emb: bool = False,#是否返回基因嵌入的标志。
        depth_mult: Optional[Tensor] = None,#深度缩放因子，形状为 (minibatch)，可选。
        do_sample: bool = False,#是否对表达水平进行采样的标志。
        do_mvc: bool = False,
        do_class: bool = False,
        get_attention_layer: list = [],#要返回哪些注意力层的列表。
    ):
        """
        forward also called on self(), a full forward pass on the model

        Args:
            gene_pos (Tensor): A tensor of shape (minibatch, seq_len)
                representing the genes used for each cell in the minibatch.
            expression (Tensor, optional): A tensor of shape (minibatch, seq_len)
                representing the expression levels of genes in the minibatch. Defaults to None.
            mask (Tensor, optional): A tensor of shape (minibatch, seq_len)
                used to mask certain elements in the sequence during the forward pass. Defaults to None.
            req_depth (Tensor, optional): A tensor of shape (minibatch,)
                representing the full depth of each sequence in the minibatch. Defaults to None.
            depth_mult (Tensor, optional): A tensor of shape (minibatch,)
                representing the depth multiplier for each sequence in the minibatch. Defaults to None.
            timepoint (Tensor, optional): A tensor of shape (minibatch,)
                representing the timepoint associated with each sequence in the minibatch. Defaults to None.
            get_gene_emb (bool, optional): A flag indicating whether to return the gene embeddings.
                If True, the gene embeddings are included in the output. Defaults to False.
            do_sample (bool, optional): A flag indicating whether to sample the expression levels.
                If True, the expression levels are sampled during the forward pass. Defaults to False.
            get_attention_layer (list, optional): A list indicating which attention layers to return.
                If not empty, the specified attention layers are included in the output. Defaults to [].

        Returns:
            dict of output Tensors: A dictionary containing the output tensors from the forward pass.
                The keys of the dictionary depend on the input flags (get_gene_emb, do_sample, get_attention_layer).res 是一个字典，包含以下键（具体键取决于输入标志）：
                at minima, the dictionary codntains the following:
                - "mean": the mean expression levels 基因表达的均值。
                - "zero_logits": the logits for zero-inflated expression levels 零膨胀表达值的 logits。
                - "disp": the dispersion parameter 分散参数。
                - "cell_embs": the cell embeddings per class 每个类别的细胞嵌入。
                - "cell_emb": the main cell embedding 主要细胞嵌入
                - "cls_output": the output of the classifier 分类器的输出。
        """
        encoding = self._encoder(#前缀的单下划线 _ 表示这个属性或方法是“受保护的”，这种命名方式常用于封装（Encapsulation），以隐藏类的内部实现细节，避免外部代码误用或意外修改。_encoder 方法返回编码后的嵌入向量 encoding。
            gene_pos,
            expression,
            mask,
            req_depth=req_depth if self.depth_atinput else None,
            timepoint=timepoint,
        )
        if self.attn_bias != "none":#注意力偏置（Attention Bias）
            if not hasattr(self, "nbias"):
                bias_path = os.path.join(
                    Path(FILEDIR).parent.parent, "data", "bias_sparse.npz"
                )
                self.nbias = torch.Tensor(load_npz(bias_path).todense()).to(
                    device=gene_pos.device, dtype=torch.float16
                )#如果没有加载偏置矩阵 nbias，从文件中加载并转换为张量。
            num = self.cell_embs_count if not self.cell_transformer else 0
            bias = torch.zeros(#初始化一个零矩阵 bias，形状为 (minibatch, seq_len + num, seq_len + num)。
                (
                    gene_pos.shape[0],
                    gene_pos.shape[1] + num,
                    gene_pos.shape[1] + num,
                ),
                device=gene_pos.device,
                dtype=torch.float16,
            )
            # fade slowly through the iterations  计算衰减因子 fade_factor，用于逐渐减少偏置的影响。
            fade_factor = 400 / (400 + self.trainer.global_step)
            # bias[:, num:, :num] = -10_000  # do not pay attention to the cls embeddings
            bias[:, num:, num:] = (
                self.nbias[gene_pos[:, :, None], gene_pos[:, None, :]] * fade_factor
            )#将偏置矩阵 nbias 应用到 bias 的对应位置，并乘以衰减因子。
        if self.cell_transformer:#Cell-specific Transformer 如果启用了self.cell_transformer,这也是在定义的时候直接初始化的一个tranformer，将编码分为细胞嵌入和基因嵌入。
            cell_encoding = encoding[:, : self.cell_embs_count, :]#encoding形状为 (minibatch, seq_len + n_labels, embsize) 的张量 提取细胞嵌入部分（cell_encoding）。
            encoding = encoding[:, self.cell_embs_count :, :]#提取基因嵌入部分（encoding）。
        transformer_output = self.transformer(#在一开始定义的时候已经初始化的一个transfoemer,输入编码后的嵌入向量 encoding。返回 Transformer 的输出 transformer_output。
            encoding,
            return_qkv=get_attention_layer,#如果 get_attention_layer 不为空，返回 QKV（查询、键、值）。
            bias=bias if self.attn_bias != "none" else None,#如果启用了注意力偏置，传递偏置矩阵 bias。
            bias_layer=list(range(self.nlayers - 1)),
        )
        if len(get_attention_layer) > 0:
            transformer_output, qkvs = transformer_output#如果需要返回注意力层的输出，将 Transformer 的输出分为 transformer_output 和 qkvs。
        if self.cell_transformer:#调用cell_transformer
            cell_output = self.cell_transformer(cell_encoding, x_kv=transformer_output)#输入细胞嵌入部分（cell_encoding）和基因嵌入 transformer_output。
            transformer_output = torch.cat([cell_output, transformer_output], dim=1)#将 Transformer 输出 cell_output 和基因嵌入 transformer_output 拼接在一起。相当于scGPT的Gene Prompt和Cell Prompt
        # if not provided we will mult by the current expression sum 
        depth_mult = expression.sum(1) if depth_mult is None else depth_mult#如果未提供深度缩放因子（depth_mult），则使用基因表达值的总和作为深度缩放因子。
        res = self._decoder(#_decoder 方法返回解码后的结果 res。
            transformer_output,
            depth_mult,
            get_gene_emb,
            do_sample,
            do_mvc,
            do_class,
            req_depth=req_depth if not self.depth_atinput else None,
        )
        return (res, qkvs) if len(get_attention_layer) > 0 else res#如果 get_attention_layer 不为空，返回一个元组 (res, qkvs)。否则，只返回 res。

    def configure_optimizers(self):#配置优化器和学习率调度器
        """@see pl.LightningModule"""
        # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        # not working because of poor weight decay implem
        if self.optim == "adam":#根据 self.optim 的值选择优化器。
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.weight_decay,
                amsgrad=False,
                fused=self.fused_adam,
            )
        elif self.optim == "adamW":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.weight_decay,
                amsgrad=False,
                fused=self.fused_adam,
            )
        elif self.optim == "galore":
            raise NotImplementedError("Galore optimizer not implemented")
            # param_groups = [
            #    {
            #        "params": [
            #            v for k, v in self.named_parameters() if "transformer" not in k
            #        ]
            #    },
            #    {
            #        "params": [
            #            v for k, v in self.named_parameters() if "transformer" in k
            #        ],
            #        "rank": 128,
            #        "update_proj_gap": 200,
            #        "scale": 0.25,
            #        "proj_type": "std",
            #    },
            # ]
            # optimizer = GaLoreAdamW(param_groups, lr=self.hparams.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optim}")
        if self.lr_reduce_monitor is None:#如果启用了学习率调整（self.lr_reduce_monitor 不为 None），配置学习率调度器。
            print("no lr reduce factor")
            return [optimizer]
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(#初始化 ReduceLROnPlateau 学习率调度器。
            optimizer,
            mode="min",
            patience=self.lr_reduce_patience,
            factor=self.lr_reduce_factor,
            verbose=True,
        )
        lr_dict = {#学习率调整器的配置 储存到字典中
            "scheduler": lr_scheduler,#学习率调度器实例。
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",#调度器的更新间隔，设置为 "epoch"，表示每个 epoch 更新一次。
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,#更新频率，设置为 1，表示每个 epoch 更新一次。
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": self.lr_reduce_monitor,#监控的指标，从 self.lr_reduce_monitor 获取。
        }
        self.lrfinder_steps = 0#检查是否启用了学习率查找器（Learning Rate Finder），并获取其训练步数。
        for val in self.trainer.callbacks:#遍历 self.trainer.callbacks，检查是否存在 _LRCallback 或 LearningRateFinder。
            if type(val) is _LRCallback:
                self.lrfinder_steps = val.num_training
            if type(val) is LearningRateFinder:
                self.lrfinder_steps = val._num_training_steps#如果找到，将 self.lrfinder_steps 设置为对应的训练步数。
        return [optimizer], [lr_dict]#返回优化器实例和学习率调度器的配置。

    def on_fit_start(self):#在训练开始之前执行一些初始化操作。
        """@see pl.LightningModule"""
        if type(self.transformer) is FlashTransformer:#如果使用的 Transformer 是 FlashTransformer，则配置其并行化设置。
            for encoder_layers in self.transformer.blocks:
                encoder_layers.set_seq_parallel(True)#调用 set_seq_parallel(True) 方法，启用序列并行化（sequence parallelization）。序列并行化：这是一种优化技术，可以提高 Transformer 在某些硬件上的性能，特别是在处理长序列时。
        for k, v in self.mat_labels_hierarchy.items():
            self.mat_labels_hierarchy[k] = v.to(self.device)#将标签层次结构矩阵（self.mat_labels_hierarchy）移动到当前设备上。

    def training_step(#定义训练循环。PyTorch Lightning 框架中 LightningModule 的一个标准方法。training_step 方法在每个训练批次（batch）中被调用，负责计算损失并记录训练指标。
        self,
        batch: Dict[str, Tensor],#batch：一个字典，包含当前批次的数据。字典的键是数据的名称，值是对应的张量。
        batch_idx,#batch_idx：当前批次的索引。
    ):
        """
        training_step defines the train loop. It is independent of forward 定义训练循环 与前向传播无关

        @see pl.LightningModule

        Returns:
            _type_: _description_
        """
        total_loss, losses = self._full_training(#调用 _full_training 方法，计算总损失和各个子任务的损失。
            batch=batch,
            do_denoise=self.do_denoise,
            noise=self.noise,
            do_next_tp=self.do_next_tp,
            do_cce=self.do_cce,
            cce_temp=self.cce_temp,
            do_ecs=self.do_ecs,
            do_mvc=self.do_mvc,
            do_adv_cls=self.do_adv_cls,
            do_adv_batch=self.do_adv_batch,
            do_cls=self.do_cls,
            do_generate=self.do_generate,
            run_full_forward=self.run_full_forward,
            mask_ratio=self.mask_ratio,
        )

        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)#log 方法记录总损失。
        self.log_dict(losses, prog_bar=True, sync_dist=True)#log_dict 方法记录各个子任务的损失。
        return total_loss
        #这种设计使得模型在每个训练批次中能够执行多种训练任务，并记录详细的训练指标，便于监控和调试。

    def _full_training(#实现了训练步骤的具体逻辑，包括前向传播、损失计算和多种训练任务。
        self,
        batch: Dict[str, Tensor],
        do_denoise: bool = False,
        noise: list[float] = [],#用于去噪的噪声级别列表。
        do_next_tp: bool = False,#指示是否执行下一次时间点预测的标志。
        do_cce: bool = False,#是否执行对比性细胞嵌入任务
        cce_temp: float = 0.5,#对比性细胞嵌入任务的相似度阈值
        do_ecs: bool = False,#是否执行弹性细胞相似性任务
        do_mvc: bool = False,#是否执行多视图编码任务
        do_adv_cls: bool = False,#是否执行对抗性分类任务
        do_adv_batch: bool = False,#是否执行对抗性批处理任务
        do_cls: bool = False,#是否执行分类任务
        do_generate: bool = False,#是否执行数据生成任务
        run_full_forward: bool = True,#是否执行完整的前向传播
        mask_ratio: list[float] = [0.15],#掩码比例列表，默认为 [0.15]。同scGPT
    ):
        """
        _full_training implement the trainng steps: forward (multiple sometimes), loss  实现训练步骤：前进，损失  为啥没有反向优化？

        Args:
            batch (dict[Tensors]): A dictionary containing tensors for the training batch:包含训练批的张量的字典
                - "x": the expression levels of genes in the minibatch 小批量中基因的表达水平
                - "genes": the genes used for each cell in the minibatch 用于小批中每个细胞的基因
                - "class": the class to predict for each cell 要预测每个细胞的类别
                - "depth": the full depth of each cell in the minibatch minibatch中每个cell的完整深度
            do_denoise (bool, optional): A flag to indicate whether to perform denoising. Defaults to False.
            noise (list[float], optional): A list of noise levels to be used in denoising. Defaults to [].
            do_next_tp (bool, optional): A flag to indicate whether to perform next time point prediction. Defaults to False.
            do_cce (bool, optional): A flag to indicate whether to perform cross-categorical entropy. Defaults to False.
            cce_temp (float, optional): The similarity threshold for cross-categorical entropy. Defaults to 0.5.
            do_ecs (bool, optional): A flag to indicate whether to perform elastic cell similarity. Defaults to False.
            do_mvc (bool, optional): A flag to indicate whether to perform multi-view coding. Defaults to False.
            do_adv_cls (bool, optional): A flag to indicate whether to perform adversarial classification. Defaults to False.
            do_generate (bool, optional): A flag to indicate whether to perform data generation. Defaults to False.
            mask_ratio (list, optional): A list of mask ratios to be used in the training. Defaults to [0.15].

        Returns:
            loss, losses: the total loss as float and the individual losses as dict
        """
        if type(mask_ratio) is not list:
            mask_ratio = [mask_ratio] #将其转换为列表
        # dynamically change the context length every 5 steps 如果启用了动态上下文长度（self.var_context_length），每 5 步随机调整一次上下文长度。
        if self.var_context_length and self.trainer.global_step % 5 == 0:
            context_length = torch.randint(400, batch["x"].shape[1], (1,)).item()
        else:
            context_length = batch["x"].shape[1]
        expression = batch["x"][:, :context_length]#提取批次中的基因表达值（expression）、基因位置（gene_pos）、总深度（total_count）、类别（clss）和批次索引（batch_idx）。
        gene_pos = batch["genes"][:, :context_length]
        total_count = batch["depth"]
        clss = batch.get("class", None)
        batch_idx = batch.get("dataset", None)

        total_loss = 0#初始化损失值
        losses = {}#初始化初始化损失列表
        cell_embs = []#初始化和嵌入列表
        if run_full_forward:#执行完整的前向传播
            output = self.forward(#调用 self.forward 方法，执行前向传播，获取输出 output。
                gene_pos,
                expression,
                mask=None,
                req_depth=total_count,
                do_mvc=do_mvc,
                do_class=do_cls,
            )
            if "disp" in output:#移除 output 中的 disp、zero_logits 和 mean 键。
                output.pop("disp")
            if "zero_logits" in output:
                output.pop("zero_logits")
            if "mean" in output:
                output.pop("mean")
            l, tot = self._compute_loss(#调用 _compute_loss 方法，计算损失 l 和总损失 tot。
                output,
                expression,
                clss,
                batch_idx,
                do_ecs,
                do_adv_cls & do_cls,
                do_adv_batch & do_cls,
            )
            cell_embs.append(output["cell_emb"].clone())#将细胞嵌入 output["cell_emb"] 添加到 cell_embs 列表中。
            full_cell_embs = output["cell_embs"].clone()#将完整的细胞嵌入 output["cell_embs"] 复制到 full_cell_embs。
            total_loss += tot#更新总损失 total_loss。
            losses.update({"full_forward_" + k: v for k, v in l.items()})#更新损失字典 losses。
            do_mvc = False
            do_cls = False

        for i in mask_ratio:
            # do noise and mask 之后再计算损失
            if do_denoise:
                expr = utils.downsample_profile(expression, dropout=0.5, randsamp=True)#如果启用了去噪任务，调用 utils.downsample_profile 方法对表达值进行降采样。
            else:
                expr = expression
            if i == "TF":
                mask = self.tf_masker(#进行加权掩码
                    ids=gene_pos,
                    mask_ratio=0.3,
                ).to(gene_pos.device)
            else:
                mask = simple_masker(#否则根据提供的值进行掩码
                    shape=gene_pos.shape,
                    mask_ratio=i,
                ).to(gene_pos.device)
            output = self.forward(
                gene_pos,
                expression=expr,
                mask=mask,
                req_depth=expr.sum(1),
                do_mvc=do_mvc,
                do_class=do_cls,
            )
            l, tot = self._compute_loss(
                output,
                expr,
                clss,
                batch_idx,
                do_ecs,
                do_adv_cls & do_cls,
                do_adv_batch & do_cls,
                do_mse=self.zinb_and_mse,
            )
            # we only want to do them once
            do_mvc = False
            do_cls = False

            cell_embs.append(output["cell_emb"].clone())
            total_loss += tot
            pct = str(int(i * 100)) + "%_" if i != "TF" else "TF_"
            losses.update({"mask_" + pct + k: v for k, v in l.items()})#更新损失列表
        # TASK 3. denoising 执行去噪任务
        if do_denoise:
            for i in noise:
                expr = utils.downsample_profile(expression, dropout=i)#降采样
                output = self.forward(
                    gene_pos,
                    expression=expr,
                    mask=None,
                    depth_mult=expression.sum(1),
                    req_depth=total_count,
                    do_mvc=do_mvc,
                    do_class=do_cls,
                )
                l, tot = self._compute_loss(
                    output,
                    expression,
                    clss,
                    batch_idx,
                    do_ecs,
                    do_adv_cls & do_cls,
                    do_adv_batch & do_cls,
                    do_mse=self.zinb_and_mse,
                )
                do_mvc = False
                do_cls = False

                cell_embs.append(output["cell_emb"].clone())
                total_loss += tot
                losses.update(
                    {"denoise_" + str(int(i * 100)) + "%_" + k: v for k, v in l.items()}
                )
                # make sure that the cell embedding stay the same even if the expression is decreased

        # TASK 6. expression generation 执行数据生成任务
        if do_generate:
            output = self._generate(#这个方法前面好像没有
                cell_embs=output["cell_embs"]
                if not run_full_forward
                else full_cell_embs,
                gene_pos=gene_pos,
                depth_mult=expression.sum(1),
                req_depth=total_count,
                do_mvc=do_mvc,
                do_class=do_cls,
            )
            if "cell_emb" in output:
                cell_embs.append(output["cell_emb"].clone())
            l, tloss = self._compute_loss(
                output,
                expression,
                clss,
                batch_idx,
                ("cell_emb" in output) and do_ecs,
                do_adv_cls & do_cls,
                do_adv_batch & do_cls,
                do_mse=self.zinb_and_mse,
            )
            losses.update({"gen_" + k: v for k, v in l.items()})
            total_loss += tloss

        # TASK 7. next time point prediction
        if do_next_tp:
            pass
        # TASK 4. contrastive cell embedding 执行交叉类别熵任务
        if do_cce:
            loss_cce = 0
            n_pairs = 0
            for i, cell_emb1 in enumerate(cell_embs[:-1]):#遍历 cell_embs 列表中的嵌入对。
                for cell_emb2 in cell_embs[(i + 1) :]:
                    loss_cce += loss.contrastive_loss(#调用 loss.contrastive_loss 方法，计算对比损失。
                        cell_emb1, cell_emb2, cce_temp
                    )  # (nlabels, minibatch, minibatch)
                    n_pairs += 1
            avg_loss_cce = loss_cce / max(n_pairs, 1)#计算平均对比损失。
            total_loss += avg_loss_cce * self.cce_scale
            # TASK 3b. contrastive graph embedding
            losses.update({"cce": avg_loss_cce})

        # TASK 8. KO profile prediction
        # if we have that information
        # TASK 9. PDgrapher-drug-like perturbation prediction (L1000?)
        return total_loss, losses#返回总损失和各个子任务的损失字典。就是多做一件事情就前向传播一次，计算损失

    def _compute_loss(#计算模型在前向传播后产生的损失。该方法支持多种任务的损失计算，包括重构掩码表达值、类别预测、对抗性分类、弹性细胞相似性等。
        self,
        output,#包含前向传播输出的字典。
        expression,#包含基因表达水平的张量。
        clss,#包含每个单元的类类的张量。
        batch_idx,
        do_ecs=False,#表示是否执行弹性细胞相似的标志。
        do_adv_cls=False,#指示是否执行对抗性分类的标志。
        do_adv_batch=False,#
        do_mse=0,
    ):
        """
        _compute_loss compute the loss of the model given output from the forward pass 计算给定前向传播的输出的模型损失

        Args:
            output (dict): A dictionary containing the output of the forward pass.
            expression (Tensor): A tensor containing the expression levels of genes.
            mask (Tensor): A tensor indicating the masked positions in the input data.
            clss (Tensor): A tensor containing the class classes for each cell.
            do_ecs (bool, optional): A flag to indicate whether to perform elastic cell similarity.
                Defaults to False.
            do_adv_cls (bool, optional): A flag to indicate whether to perform adversarial classification.
                Defaults to False.
            do_mse (float, optional): A scaling factor to indicate whether and how much to weight mean
            squared error loss in addition to zinb loss.
                Defaults to 0.

        Raises:
            ValueError: Raised when an invalid operation or input is encountered.

        Returns:
            tuple: A tuple containing the total loss as a float and the individual losses as a dictionary.包含总损失为浮点数和单个损失为字典的元组。
        """
        total_loss = 0#初始化总损失和损失字典
        losses = {}
        # TASK 1. reconstruct masked expression 重构掩码表达值的损失
        if "zero_logits" in output:#如果输出中包含 zero_logits，使用零膨胀负二项分布（ZINB）损失。
            loss_expr = loss.zinb(
                theta=output["disp"],
                pi=output["zero_logits"],
                mu=output["mean"],
                target=expression,
            )
            if do_mse:#如果启用了均方误差（MSE）损失，将其加到总损失中。
                loss_expr += (
                    loss.mse(
                        input=torch.log(output["mean"] + 1)
                        * (1 - torch.sigmoid(output["zero_logits"])),
                        target=torch.log(expression + 1),
                    )
                    / 10  # scale to make it more similar to the zinb
                )
        elif "disp" in output:#如果输出中包含 disp，使用负二项分布（NB）损失。
            loss_expr = loss.nb(
                theta=output["disp"],
                mu=output["mean"],
                target=expression,
            )
        elif "mean" in output:#如果输出中包含 mean，使用均方误差损失。
            loss_expr = loss.mse(
                input=output["mean"],
                target=expression,
            )
        else:
            loss_expr = 0
        total_loss += loss_expr#将计算的表达损失加到 total_loss 中
        losses.update({"expr": loss_expr})#更新 losses 字典

        # TASK 2. predict classes 类别预测的损失
        if len(self.classes) > 0 and "cell_embs" in output:#如果模型有类别标签，并且输出中包含细胞嵌入，计算嵌入的独立性损失。
            ## Calculate pairwise cosine similarity for the embeddings 计算嵌入的两两余弦相似度
            # Calculate pairwise cosine similarity more efficiently
            loss_emb_indep = loss.within_sample(output["cell_embs"])
            losses.update({"emb_independence": loss_emb_indep})
            total_loss += self.class_embd_diss_scale * loss_emb_indep
            ## compute class loss
            loss_cls = 0
            loss_adv_cls = 0
            for j, clsname in enumerate(self.classes):#遍历所有类别，计算分类损失。
                if "cls_output_" + clsname not in output:
                    continue
                # setting the classes from index to one hot
                loss_cls += loss.classification(
                    clsname,
                    pred=output["cls_output_" + clsname],
                    cl=clss[:, j],
                    maxsize=self.label_counts[clsname],
                    labels_hierarchy=self.mat_labels_hierarchy,
                )
            total_loss += self.class_scale * loss_cls
            if loss_cls != 0:
                losses.update({"cls": loss_cls})
            # TASK 2bis. adversarial label prediction
            if do_adv_cls:#如果启用了对抗性分类，计算对抗性分类损失。
                embs = output["cell_embs"][
                    :, (2 if self.depth_atinput else 1) :, :
                ].clone()
                for j, adv_cls in enumerate(self.classes):
                    ind = torch.arange(len(self.classes))
                    mean_embs = torch.mean(embs[:, ind != j, :], dim=1)
                    mean_embs = grad_reverse(mean_embs, lambd=1.0)
                    adv_pred = self.cls_decoders[adv_cls](mean_embs)
                    loss_adv_cls += loss.classification(
                        adv_cls,
                        pred=adv_pred,
                        cl=clss[:, j],
                        maxsize=self.label_counts[adv_cls],
                        labels_hierarchy=self.mat_labels_hierarchy,
                    )

                total_loss += self.adv_class_scale * loss_adv_cls
                losses.update({"adv_cls": loss_adv_cls})

        if (#对抗性批次任务的损失
            do_adv_batch
            and self.grad_reverse_discriminator_loss is not None
            and batch_idx is not None
            and "cell_embs" in output
        ):
            mean_emb = torch.mean(
                output["cell_embs"][:, (2 if self.depth_atinput else 1) :, :].clone(),
                dim=1,
            )
            loss_adv = self.grad_reverse_discriminator_loss(mean_emb, batch_idx)
            total_loss += loss_adv * self.class_scale / 16
            losses.update({"adv_batch": loss_adv})
        # TASK 2ter. cell KO effect prediction
        # (just use a novel class, cell state and predict if cell death or not from it)
        # add large timepoint and set the KO gene to a KO embedding instead of expression embedding
        # TODO: try to require the gene id to still be predictable (with weight tying)
        if "mvc_zero_logits" in output:#计算多视图编码的损失。如果输出中包含 mvc_zero_logits，使用 ZINB 损失。
            loss_expr_mvc = loss.zinb(
                theta=output["mvc_disp"],
                pi=output["mvc_zero_logits"],
                mu=output["mvc_mean"],
                target=expression,
            )
            total_loss += loss_expr_mvc * self.mvc_scale
            losses.update({"expr_mvc": loss_expr_mvc})
        elif "mvc_mean" in output:#如果输出中包含 mvc_mean，使用 MSE 损失。
            loss_expr_mvc = loss.mse(
                input=output["mvc_mean"],
                target=expression,
            )
            total_loss += loss_expr_mvc * self.mvc_scale
            losses.update({"expr_mvc": loss_expr_mvc})
        # TASK 5. elastic cell similarity 弹性细胞相似性的损失
        if do_ecs and "cell_emb" in output:
            loss_ecs = loss.ecs(output["cell_emb"], ecs_threshold=self.ecs_threshold)
            total_loss += self.ecs_scale * loss_ecs
            losses.update({"ecs": loss_ecs})
        return losses, total_loss#返回总损失和损失字典

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):#于自定义优化器的步骤。这个方法在每次优化器更新时被调用，通常用于执行学习率预热（warm-up）等操作。
        """@see pl.LightningModule"""
        # update params epoch：当前的训练轮数。 batch_idx：当前批次的索引。optimizer：优化器实例。 optimizer_closure：一个闭包函数，用于重新计算损失并返回梯度。
        # manually warm up lr without a scheduler
        # making sure that we don't do this during lrfinder
        lr_scale = None#学习率预热（Learning Rate Warm-up）
        prev_lr = None
        if (
            self.trainer.global_step < self.warmup_duration + self.lrfinder_steps
        ) and self.lrfinder_steps <= self.trainer.global_step:#检查当前全局步数是否在预热阶段（self.trainer.global_step < self.warmup_duration + self.lrfinder_steps）。
            for i, pg in enumerate(optimizer.param_groups):#遍历优化器的参数组，更新每个参数组的学习率。
                lr_scale = min(
                    1.0, float(self.trainer.global_step + 1) / self.warmup_duration
                )#如果在预热阶段，计算学习率的缩放因子 lr_scale，其值在 [0, 1] 之间，随着步数增加而逐渐接近 1。
                prev_lr = pg["lr"]#保存每个参数组的原始学习率 prev_lr，以便在需要时恢复。
                pg["lr"] = lr_scale * self.hparams.lr
        for i, pg in enumerate(optimizer.param_groups):#遍历优化器的参数组，使用 self.log 方法记录每个参数组的学习率。
            # if pg["lr"] < 2e-5:
            #    pg["lr"] = 2e-5
            self.log("lr_" + str(i), pg["lr"])
        if optimizer.param_groups[0]["lr"] > self.hparams.lr:#检查优化器的学习率是否超过了预设的学习率。
            print(optimizer.param_groups[0]["lr"], self.hparams.lr)
            print(lr_scale, self.warmup_duration, self.trainer.global_step, prev_lr)#如果优化器的第一个参数组的学习率超过了预设的学习率（self.hparams.lr），打印相关信息。
            if prev_lr is not None:#如果 prev_lr 不为 None，将学习率恢复到之前的值。
                pg["lr"] = prev_lr
            else:
                raise ValueError("OPTIMIZER HAS INCREASED LR. WHYY?")#如果 prev_lr 为 None，抛出一个错误，提示学习率异常增加。

        optimizer.step(closure=optimizer_closure)#调用 optimizer.step 方法，执行优化器的更新步骤。使用 optimizer_closure 重新计算损失并返回梯度。

    def on_validation_start(self):#on_validation_start 在验证阶段开始之前被调用，通常用于执行一些初始化操作。
        for k, v in self.mat_labels_hierarchy.items():#将标签层次结构矩阵（self.mat_labels_hierarchy）移动到当前设备上。
            self.mat_labels_hierarchy[k] = v.to(self.device)

    def on_validation_epoch_start(self):#on_validation_epoch_start 在每个验证轮次（epoch）开始之前被调用，通常用于执行一些初始化操作。
        self.embs = None#初始化嵌入变量
        self.counter = 0#初始化计数器

    def validation_step(#用于定义验证循环
        self,
        batch,
        batch_idx,
    ):
        """
        validation_step defines the validation loop. It is independent of forward
        @see pl.LightningModule 是 PyTorch Lightning 框架中 LightningModule 的一个标准方法

        Args:
            batch (list[Tensor]): @see training_step
        """
        val_loss, losses = self._full_training(#调用 _full_training 方法计算损失
            batch=batch,
            do_denoise=self.do_denoise,
            noise=self.noise,
            do_next_tp=self.do_next_tp,
            do_cce=self.do_cce,
            cce_temp=self.cce_temp,
            do_ecs=self.do_ecs,
            do_mvc=self.do_mvc,
            do_adv_cls=self.do_adv_cls,
            do_adv_batch=self.do_adv_batch,
            do_cls=self.do_cls,
            do_generate=self.do_generate,
            run_full_forward=self.run_full_forward,
            mask_ratio=self.mask_ratio,
        )
        expression = batch["x"]#从批次中提取基因表达值、基因位置和深度信息。
        gene_pos = batch["genes"]
        depth = batch["depth"]
        # TODO: make this faster by only calling val loss 
        if self.embs is not None: #根据条件执行预测，并更新嵌入信息。
            if self.embs.shape[0] < 100_000:
                self.info = torch.cat([self.info, batch["class"]])#如果 self.embs 不为 None，并且其大小小于 100,000，将当前批次的类别信息追加到 self.info 中。
                self._predict(
                    gene_pos,
                    expression,
                    depth,
                    pred_embedding=self.pred_embedding,
                    max_size_in_mem=120_000,
                )#调用 _predict 方法，执行预测操作。
        else:
            self.info = batch["class"]#如果 self.embs 为 None，直接将当前批次的类别信息赋值给 self.info，并调用 _predict 方法。
            self._predict(
                gene_pos,
                expression,
                depth,
                pred_embedding=self.pred_embedding,
                max_size_in_mem=120_000,
            )
        self.log("val_loss", val_loss, sync_dist=True)#log 方法记录验证损失。
        self.log_dict(losses, sync_dist=True)
        return val_loss
    #使得模型在每个验证批次中能够执行多种任务，并记录详细的验证指标，便于监控和调试

    def on_validation_epoch_end(self):#on_validation_epoch_end 在每个验证轮次（epoch）结束时被调用，通常用于执行一些清理或日志记录操作。
        """@see pl.LightningModule PyTorch Lightning 框架中 LightningModule 的一个钩子方法。"""
        self.embs = self.all_gather(self.embs).view(-1, self.embs.shape[-1])#self.embs、self.info、self.pred 和 self.pos 是在验证过程中收集的嵌入、类别信息、预测结果和位置信息。
        self.info = self.all_gather(self.info).view(-1, self.info.shape[-1])#使用 self.all_gather 方法将这些信息从所有进程聚合到一起。
        self.pred = (
            self.all_gather(self.pred).view(-1, self.pred.shape[-1])
            if self.pred is not None
            else None#如果 self.pred 为 None，则保持其为 None。
        )
        self.pos = self.all_gather(self.pos).view(-1, self.pos.shape[-1])#调整聚合后的张量的形状，使其成为二维张量，形状为 (total_samples, feature_dim)。
        if self.trainer.state.stage != "sanity_check":#检查当前是否处于验证阶段，而不是进行 sanity check（完整性检查）。如果当前阶段不是 sanity check，则执行后续操作。
            if self.trainer.is_global_zero:#在主进程（is_global_zero）中记录日志并更新学习率调度器。
                print("logging anndata")
                sch = self.lr_schedulers()#获取学习率调度器（self.lr_schedulers()）
                sch.step(self.trainer.callback_metrics["val_loss"])#调用 step 方法，根据验证损失（self.trainer.callback_metrics["val_loss"]）更新学习率。
                # run the test function on specific dataset
                self.log_adata(
                    gtclass=self.info, name="validation_part_" + str(self.counter)
                )#调用 self.log_adata 方法，记录验证数据，其中 gtclass 是真实类别信息，name 是日志的名称。
                if (self.current_epoch + 1) % self.test_every == 0:
                    self.on_test_epoch_end()#如果当前轮次是测试间隔的倍数（self.test_every），调用 self.on_test_epoch_end 方法。
                # Synchronize all processes with a timeout
            if torch.distributed.is_initialized():#检查是否初始化了分布式训练（torch.distributed.is_initialized()）。
                # Set a timeout that's longer than your test typically takes
                # Write rank to file for debugging
                self.trainer.strategy.barrier()#如果是，调用 self.trainer.strategy.barrier() 方法，同步所有进程。
    #使得模型在每个验证轮次结束时能够正确地聚合数据、记录日志、更新学习率，并同步所有进程，从而提高验证过程的效率和稳定性。

    def test_step(self, *args, **kwargs):
        pass

    def on_test_epoch_end(self):#on_test_epoch_end 在每个测试轮次（epoch）结束时被调用，通常用于执行测试相关的操作，如评估模型性能、记录测试指标等。
        # Run the test only on global rank 0 确保测试操作仅在主进程（global rank 0）中执行。
        name = self.name + "_step" + str(self.global_step)#定义测试结果的名称，包含模型名称和当前全局步数。
        try:
            metrics = utils.test(self, name, filedir=str(FILEDIR), do_class=self.do_cls)#调用 utils.test 方法，传入当前模型实例、测试结果的名称、文件目录和是否执行分类任务的标志。
            print(metrics)#utils.test 方法通常会执行模型的测试逻辑，返回一个包含测试指标的字典 metrics。
            print("done test")
            self.log_dict(metrics, sync_dist=False, rank_zero_only=True)#使用 self.log_dict 方法记录测试指标，设置 sync_dist=False 和 rank_zero_only=True，确保仅在主进程中记录日志。
        except Exception as e:#捕获并处理测试过程中可能发生的异常。
            import traceback

            print(f"Error during test: {e}")#捕获任何异常 e。
            print("Full traceback:")#打印错误信息和完整的堆栈跟踪。
            print(traceback.format_exc())
            print("Skipping test metrics logging")
    #使得模型在每个测试轮次结束时能够正确地执行测试逻辑、记录测试指标，并处理可能出现的异常

    def on_predict_epoch_start(self):#on_predict_epoch_start 在每个预测轮次（epoch）开始之前被调用，通常用于执行一些初始化操作。
        """@see pl.LightningModule PyTorch Lightning 框架中 LightningModule 的一个钩子方法。"""
        self.embs = None#初始化嵌入变量
        self.attn.data = None#初始化注意力模块的数据
        self.attn.attn = None
        self.counter = 0#初始化计数器
        if type(self.transformer) is FlashTransformer:#如果使用的 Transformer 是 FlashTransformer，配置其并行化设置。
            for encoder_layers in self.transformer.blocks:
                encoder_layers.set_seq_parallel(False)
    #使得模型在每个预测轮次开始之前能够正确地初始化相关变量，避免数据混淆或计数错误

    def predict_step(self, batch, batch_idx):#定义预测循环。predict_step 方法在每个预测批次（batch）中被调用，负责对输入数据进行预测并返回结果。
        """
        embed given gene expression, encode the gene embedding and cell embedding.对给定的基因表达数据进行嵌入，生成基因嵌入和细胞嵌入。

        Args:
            batch @see training_step PyTorch Lightning 框架中 LightningModule 的一个标准方法

        Returns:
            Tensor: _description_ 返回预测结果，通常是一个张量（Tensor）。
        """
        return self._predict(
            batch["genes"],
            batch["x"],
            batch["depth"],
            self.predict_mode,
            self.pred_embedding,
            self.get_attention_layer,
            self.predict_depth_mult,
        )

    def _predict(#用于执行预测逻辑。根据不同的预测模式（predict_mode）和配置，生成基因嵌入、细胞嵌入、预测的细胞类别等信息。
        self,
        gene_pos,
        expression,
        depth,
        predict_mode="none",
        pred_embedding=[],#预测嵌入的类别，默认为空列表
        get_attention_layer=[],#要获得注意的层。
        depth_mult=6,#深度的倍数，默认为 6
        keep_output=True,#Keep_output （bool，可选）：是否将输出保存在内存中。
        max_size_in_mem=100_000,
        get_gene_emb=False,
    ):
        """
        @see predict_step will save output of predict in multiple self variables

        - embs: the cell embeddings (means from label specific embeddings given by self.pred_embedding)细胞嵌入（表示来自self.pred_embedding给出的标签特定嵌入）
        - pred: the predicted cell classes 预测的细胞类别
        - pos: the genes used 使用的基因
        - expr_pred: the expression prediction. [mean, disp, zero_logits] 表达式预测。
        - mean_attn: the mean attention across cells for the given layer (in self.get_attention_layer) 给定层中各单元的平均注意力（在self.get_attention_layer中）

        these will be finalized in self.on_predict_epoch_end() 这些将在self.on_predict_epoch_end（）中完成。

        Args:
            @see training_step
            other important arguments:
            keep_output (bool, optional): whether to keep the output in memory. Defaults to True. 
            self.get_attention_layer (list, optional): the layers to get the attention from. Defaults to [].
            self.pred_embedding (list, optional): the classes to predict. Defaults to [].

        """
        if predict_mode == "none":#根据 predict_mode 的值，选择不同的预测逻辑。
            output = self.forward(#调用 self.forward 方法，执行前向传播。
                gene_pos,
                expression,
                depth_mult=expression.sum(1),
                req_depth=depth,
                get_attention_layer=get_attention_layer,
                do_class=True,
                get_gene_emb=get_gene_emb,
            )
            if len(get_attention_layer) > 0:#如果需要获取注意力层的输出，调用 self.attn.add 方法。
                self.attn.add([i[:, :, :2, :] for i in output[1]], gene_pos)
                output = output[0]
            cell_embs = output["cell_embs"]#提取细胞嵌入 cell_embs。
        elif predict_mode == "denoise":
            output = self.forward(
                gene_pos,
                expression,
                depth_mult=expression.sum(1) * depth_mult,
                req_depth=depth * depth_mult,
                get_attention_layer=get_attention_layer,
                do_class=True,
                get_gene_emb=get_gene_emb,
            )
            if len(get_attention_layer) > 0:
                self.attn.add([i[:, :, :2, :] for i in output[1]], gene_pos)
                output = output[0]
            cell_embs = output["cell_embs"]
        elif predict_mode == "generate":#predict_mode == "generate"：
            output = self.forward(
                gene_pos,
                expression,
                req_depth=depth,
                do_mvc=False,
                do_class=False,
                get_gene_emb=get_gene_emb,
            )
            cell_embs = output["cell_embs"]
            output = self._generate(#调用 _generate 方法，生成预测结果。
                output["cell_embs"],
                gene_pos,
                req_depth=None,  # otherwise we have 2 depths passed
                depth_mult=expression.sum(1),
                do_class=self.do_cls,
                do_mvc=False,
            )
        else:
            raise ValueError(#抛出 ValueError，提示 predict_mode 必须是 ['none', 'denoise', 'generate'] 中的一个。
                "predict_mode needs to be one of ['none', 'denoise', 'generate']"
            )

        if len(pred_embedding) == 0:#处理预测嵌入,如果 pred_embedding 为空，则使用所有类别。
            pred_embedding = self.classes
        ind = [#计算每个类别的索引，考虑是否启用了深度输入。
            self.classes.index(i) + (2 if self.depth_atinput else 1)
            for i in pred_embedding
        ]
        if not keep_output:#如果 keep_output 为 False，直接返回预测结果。
            return {#返回一个字典，包含嵌入、类别、基因位置和表达预测。
                "embs": torch.mean(cell_embs[:, ind, :], dim=1),#计算细胞嵌入的均值。
                "class": (#计算预测的细胞类别。
                    torch.stack(
                        [
                            torch.argmax(output["cls_output_" + clsname], dim=1)
                            for clsname in self.classes
                        ]
                    ).transpose(0, 1)
                    if len(self.classes) > 0
                    else None
                ),
                "pos": gene_pos,
                "expr": (
                    [output["mean"], output["disp"], output["zero_logits"]]
                    if "disp" in output
                    else [output["mean"]]
                ),
            }
        if self.embs is None:#保存预测结果到模型的属性中。如果 self.embs 为 None，初始化嵌入、预测类别、基因位置和表达预测。
            self.embs = torch.mean(cell_embs[:, ind, :], dim=1)
            # self.embs = output["cls_output_" + "cell_type_ontology_term_id"]
            self.pred = (
                torch.stack(
                    [
                        (
                            torch.argmax(output["cls_output_" + clsname], dim=1)
                            if not self.keep_all_cls_pred
                            else output["cls_output_" + clsname]
                        )
                        for clsname in self.classes
                    ]
                ).transpose(0, 1)
                if len(self.classes) > 0
                else None
            )
            self.pos = gene_pos
            self.expr_pred = (
                [output["mean"], output["disp"], output["zero_logits"]]
                if "disp" in output
                else [output["mean"]]
            )
        else:#如果 self.embs 不为 None，将当前批次的结果追加到已保存的结果中。
            self.embs = torch.cat(
                # [self.embs, output["cls_output_" + "cell_type_ontology_term_id"]]
                [self.embs, torch.mean(cell_embs[:, ind, :], dim=1)]
            )
            self.pred = torch.cat(
                [
                    self.pred,
                    (
                        torch.stack(
                            [
                                (
                                    torch.argmax(output["cls_output_" + clsname], dim=1)
                                    if not self.keep_all_cls_pred
                                    else output["cls_output_" + clsname]
                                )
                                for clsname in self.classes
                            ]
                        ).transpose(0, 1)
                        if len(self.classes) > 0
                        else None
                    ),
                ],
            )
            self.pos = torch.cat([self.pos, gene_pos])
            self.expr_pred = (
                [
                    torch.cat([self.expr_pred[0], output["mean"]]),
                    torch.cat([self.expr_pred[1], output["disp"]]),
                    torch.cat([self.expr_pred[2], output["zero_logits"]]),
                ]
                if "disp" in output
                else [torch.cat([self.expr_pred[0], output["mean"]])]
            )
        if self.embs is not None:#检查保存的预测结果是否超过内存限制。
            if self.embs.shape[0] > max_size_in_mem and self.pred_log_adata:#如果保存的嵌入数量超过 max_size_in_mem，并且 self.pred_log_adata 为 True，调用 self.log_adata 方法记录当前部分的预测结果。
                print("logging")
                self.log_adata(name="predict_part_" + str(self.counter))
                self.counter += 1
                self.pos = None
                self.expr_pred = None
                self.pred = None
                self.embs = None#清空保存的预测结果，以避免内存溢出。
            elif not self.pred_log_adata:#如果 self.pred_log_adata 为 False，打印警告信息，提示需要设置 pred_log_adata 为 True。
                print(
                    "WARNING, reached max size in memory, deleting the adata, \
                    need to set pred_log_adata to True to log the adata"
                )
    #使得模型在预测过程中能够灵活地处理不同模式的预测任务，并有效地管理内存，避免内存溢出。

    def on_predict_epoch_end(self):#on_predict_epoch_end 在每个预测轮次（epoch）结束时被调用，通常用于执行一些清理或日志记录操作。
        """@see pl.LightningModule will"""
        if self.pos.shape[0] < 100:#检查保存的预测结果的数量是否小于 100。如果 self.pos 的第一维大小（即预测结果的数量）小于 100，直接返回，不执行后续操作。
            return
        if self.pred_log_adata:#如果 self.pred_log_adata 为 True，表示需要将预测结果记录到磁盘。
            print("adding on disk")
            return self.log_adata(name="predict_part_" + str(self.counter))#返回 self.log_adata 的结果，结束方法的执行。
    #使得模型在预测过程中能够灵活地管理预测结果，避免内存溢出，并确保预测结果的完整性和可追溯性

    def get_cell_embs(self, layer_output):#从模型的某一层的输出中提取细胞嵌入（cell embeddings）
        """
        get_cell_embs

        Args:
            layer_output (Tensor): The output tensor from a layer in the model.   layer_output：模型某一层的输出张量，形状通常为 (minibatch, seq_len, embsize)。

        Raises:
            ValueError: Raised when an unknown cell embedding style is encountered.

        Returns:
            Tensor: The cell embeddings tensor. 返回细胞嵌入张量，形状通常为(minibatch, embsize)。
        """
        if self.cell_emb_style == "cls" and self.classes is not None:#如果细胞嵌入风格为 "cls"，并且模型有类别信息，则从层输出中提取细胞嵌入。
            # (minibatch, embsize)
            cell_emb = layer_output[:, : self.cell_embs_count]#提取 layer_output 的前 self.cell_embs_count 列，作为细胞嵌入。 
        elif self.cell_emb_style == "avg-pool":#如果细胞嵌入风格为 "avg-pool"，则对层输出进行平均池化，提取细胞嵌入。
            cell_emb = torch.mean(layer_output, dim=1)#使用 torch.mean 对 layer_output 的第 1 维（dim=1）进行平均池化。返回的 cell_emb 形状为 (minibatch, embsize)。
        else:
            raise ValueError(f"Unknown cell_emb_style: {self.cell_emb_style}")#如果细胞嵌入风格既不是 "cls" 也不是 "avg-pool"，抛出 ValueError。
        return cell_emb#cell_emb：细胞嵌入张量，形状为 (minibatch, embsize)。

    def _generate(
        self,
        cell_embs: Tensor,#表示细胞嵌入的张量。
        gene_pos: Tensor,
        depth_mult: Tensor,
        req_depth: Optional[Tensor] = None,
        **decoder_kwargs,
    ):
        """
        _generate given cell_embeddings, generate an expression profile 生成给定cell_embeddings的表达文件

        the goal was to iterate multiple times,  我们的目标是多次迭代，
        to create a trajectory and reach a certain state  创造一个轨迹并达到一定的状态
        should call forward multiple times  需要前向传播多次

        Args:
            cell_emb(:obj:`Tensor`): A tensor representing cell embeddings. It has a shape of (minibatch, embsize).
            src(:obj:`Tensor`): A tensor representing the source data. It has a shape of (minibatch, seq_len).  表示源数据的张量。
            values(:obj:`Tensor`): An optional tensor representing the values. It has a shape of (minibatch, seq_len).  表示值的可选张量。
            gen_iters(:obj:`int`): An integer representing the number of generation iterations.  表示生成迭代次数的整数。
            classes(:obj:`Tensor`): An optional tensor representing the classes. It has a shape of (batch,).  代表类的可选张量。
        """
        if req_depth is not None and self.depth_atinput:#如果启用了深度输入（self.depth_atinput），并且提供了深度信息（req_depth），调整细胞嵌入。
            cell_embs = torch.cat([cell_embs[:, :1, :], cell_embs[:, 2:, :]], dim=1)#从 cell_embs 中移除第二列（通常用于深度信息）。将调整后的细胞嵌入拼接回去。
        encoding = self._encoder(#调用 _encoder 方法，将细胞嵌入和基因位置信息编码为嵌入向量。变量encoding很重要
            cell_embs=cell_embs,
            gene_pos=gene_pos,
            req_depth=req_depth if self.depth_atinput else None,
        )
        if self.cell_transformer:#如果启用了细胞特定的 Transformer：
            gene_encoding = encoding[:, self.cell_embs_count :, :]#提取基因编码部分（gene_encoding）和细胞嵌入部分（cell_embs）。
            cell_embs = encoding[:, : self.cell_embs_count, :]
            transformer_output = self.transformer(gene_encoding, x_kv=cell_embs)#调用 self.transformer，将基因编码传递给 Transformer，并使用细胞嵌入作为键值对（x_kv）。
            transformer_output = torch.cat([cell_embs, transformer_output], dim=1)#将 Transformer 的输出与细胞嵌入拼接在一起。
        else:
            transformer_output = self.transformer(encoding)#如果未启用单元格特定的 Transformer，直接将编码后的嵌入向量传递给 Transformer。
        output = self._decoder(#调用 _decoder 方法，将 Transformer 的输出解码为基因表达谱。
            transformer_output,
            depth_mult=depth_mult,
            req_depth=req_depth if not self.depth_atinput else None,
            **decoder_kwargs,
        )
        if self.cell_transformer:
            output.pop("cell_embs")#从 output 中移除 cell_embs 和 cell_emb 键。
            output.pop("cell_emb")
        return output  # (minibatch, seq_len)

    def log_adata(self, gtclass=None, name=""):#将预测结果记录为一个 AnnData 对象，并将其保存到磁盘。
        """
        log_adata will log an adata from predictions. Log_adata将记录来自预测的数据。
        It will log to tensorboard and wandb if available

        see @utils.log_adata
        """
        try:#确定保存 AnnData 对象的目录。
            mdir = self.logger.save_dir if self.logger.save_dir is not None else "/tmp"#尝试从 self.logger.save_dir 获取保存目录。
        except:
            mdir = "data/"#如果 self.logger.save_dir 为 None，则使用默认目录 /tmp。
        if not os.path.exists(mdir):#如果目录不存在，创建该目录。
            os.makedirs(mdir)
        adata, fig = utils.make_adata(#调用 utils.make_adata 方法，生成 AnnData 对象和可视化图形。
            self.embs,
            self.classes,
            self.pred if not self.keep_all_cls_pred else None,
            self.attn.get(),
            self.global_step,
            self.label_decoders,
            self.labels_hierarchy,
            gtclass,
            self.name + "_" + name + "_" + str(self.global_rank),
            mdir,
            self.doplot,
        )
        if self.doplot:#将可视化图形记录到 TensorBoard 和 WandB。
            try:
                self.logger.experiment.add_figure(fig)
            except:
                print("couldn't log to tensorboard")
            try:
                self.logger.log_image(key="umaps", images=[fig])
            except:
                print("couldn't log to wandb")

        return adata
    #使得模型能够将预测结果记录为一个标准的 AnnData 对象，并将其保存到磁盘，同时支持将结果记录到 TensorBoard 和 WandB，便于后续的分析和可视化
 
    def _predict_denoised_expression(self, gene_pos, expression, depth):#用于预测去噪后的基因表达谱。
        """
        Args:
            gene_pos (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            expression (:obj:`Tensor`): token values, shape [batch_size, seq_len]

        Returns:
            dict of output Tensors.
        """
        output = self.forward(gene_pos, expression, req_depth=depth)#调用模型的 forward 方法，执行前向传播。
        return output
