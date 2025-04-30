from torch import nn
import torch.nn.functional as F
from KMGCLModels.Encoder import Encoder
from KMGCLModels.Projection import Projection
from torchmetrics.functional import pairwise_cosine_similarity
import torch

class KMGCLModel(nn.Module):
    def __init__(
        self,
        graph_model,
        cnmr_model,
        image_model,
        smiles_model,
        config,

    ):
        super().__init__()
        self.graph_encoder = Encoder(model=graph_model, trainable=True)
        self.nmr_encoder = Encoder(model=cnmr_model, trainable=False)
        self.image_encoder = Encoder(model=image_model, trainable=False)
        self.smiles_encoder = Encoder(model=smiles_model, trainable=False)
        self.graphMetric_method = config.graphMetric_method
        self.alpha = config.alpha
        self.device = config.device

    def forward(self, batch):

        # GraphEmbedding & NodeEmbedding
        # Zhengyang's modification to use Molecule Dataset
        graph = batch['smiles_input'].batch_graph()
        #print(len(graph))
        
        mask = torch.nonzero(batch['graph'].x[:, 0] == 5.0).squeeze(1)
        
        
        graphEmbedding, nodeEmbedding = self.graph_encoder(graph)
   
        nodeEmbedding = nodeEmbedding[mask] # Only extract the Carbon information
        # Zhengyang's modification ends
        graphEmbedding = F.normalize(graphEmbedding, p=2, dim=1)
        #print("graphEmbedding shape")
        #print(graphEmbedding.shape)
        nodeEmbedding = F.normalize(nodeEmbedding, p=2, dim=1)
        #print("nodeEmbedding shape")
        #print(nodeEmbedding.shape)

        # nodeMetric
        ppm_diff = batch['peak']
        #print("ppm_diff shape")
        #print(ppm_diff.shape)
        nodeMetric = F.softmax(ppm_diff, dim=-1)

        # nodeLoss
        nodeLogits = nodeEmbedding @ nodeEmbedding.T
        nodeLoss = F.cross_entropy(nodeLogits, nodeMetric) + F.cross_entropy(nodeLogits.T, nodeMetric.T)

        # graphMetric
        graphMetric = self.genGraphMetric(batch)

        # graphLoss
        graphLogits = graphEmbedding @ graphEmbedding.T
        graphLoss = F.cross_entropy(graphLogits, graphMetric) + F.cross_entropy(graphLogits.T, graphMetric.T)

        loss = self.alpha * nodeLoss + (1-self.alpha) * graphLoss
        return loss, nodeLoss, graphLoss, graphLogits

    def compute_image_metric(self, image):
        image_embeddings = self.image_encoder(image)
        image_metric = pairwise_cosine_similarity(image_embeddings,image_embeddings)
        image_metric = F.softmax(image_metric, dim=-1)
        return image_metric

    def compute_nmr_metric(self, nmr):
        nmr_embeddings = self.nmr_encoder(nmr)
        nmr_metric = pairwise_cosine_similarity(nmr_embeddings,nmr_embeddings)
        nmr_metric = F.softmax(nmr_metric, dim=-1)
        return nmr_metric

    def compute_smiles_metric(self, smiles):
        smiles_embeddings = self.smiles_encoder(smiles)
        smiles_metric = pairwise_cosine_similarity(smiles_embeddings,smiles_embeddings)
        smiles_metric = F.softmax(smiles_metric, dim=-1)
        return smiles_metric

    def compute_fingerprint_metric(self, fingerprint):
        fp_intersection = fingerprint @ fingerprint.T
        fp_sum = torch.sum(fingerprint, dim=-1)
        fp_sum = fp_sum + fp_sum.view(-1, 1)
        fingerprint_metric = fp_intersection / (fp_sum - fp_intersection)
        fingerprint_metric = F.softmax(fingerprint_metric, dim=-1)
        return fingerprint_metric

    def compute_fusion_metric(self, batch):
        nmr_metric = self.compute_nmr_metric(batch['nmr'])
        image_metric = self.compute_image_metric(batch['image'])
        smiles_metric = self.compute_smiles_metric(batch['smiles'])
        fingerprint_metric = self.compute_fingerprint_metric(batch['fingerprint'])

        fusion_metric = torch.cat([nmr_metric, image_metric, smiles_metric, fingerprint_metric], dim=1)
        fusion_metric = fusion_metric @ fusion_metric.T
        fusion_metric = F.softmax(fusion_metric, dim=-1)

        return fusion_metric
    
    def compute_fusion_metric_image(self, batch):
        nmr_metric = self.compute_nmr_metric(batch['nmr'])
        image_metric = self.compute_image_metric(batch['image'])
        smiles_metric = self.compute_smiles_metric(batch['smiles'])
        fingerprint_metric = self.compute_fingerprint_metric(batch['fingerprint'])

        fusion_metric_image = 0.7 * image_metric + 0.1 * nmr_metric + 0.1 * smiles_metric + 0.1 * fingerprint_metric

        return fusion_metric_image
    
    def compute_fusion_metric_nmr(self, batch):
        nmr_metric = self.compute_nmr_metric(batch['nmr'])
        image_metric = self.compute_image_metric(batch['image'])
        smiles_metric = self.compute_smiles_metric(batch['smiles'])
        fingerprint_metric = self.compute_fingerprint_metric(batch['fingerprint'])

        fusion_metric_nmr = 0.1 * image_metric + 0.7 * nmr_metric + 0.1 * smiles_metric + 0.1 * fingerprint_metric

        return fusion_metric_nmr
    
    def compute_fusion_metric_smiles(self, batch):
        nmr_metric = self.compute_nmr_metric(batch['nmr'])
        image_metric = self.compute_image_metric(batch['image'])
        smiles_metric = self.compute_smiles_metric(batch['smiles'])
        fingerprint_metric = self.compute_fingerprint_metric(batch['fingerprint'])

        fusion_metric_smiles = 0.1 * image_metric + 0.1 * nmr_metric + 0.7 * smiles_metric + 0.1 * fingerprint_metric

        return fusion_metric_smiles
    
    def compute_fusion_metric_fingerprint(self, batch):
        nmr_metric = self.compute_nmr_metric(batch['nmr'])
        image_metric = self.compute_image_metric(batch['image'])
        smiles_metric = self.compute_smiles_metric(batch['smiles'])
        fingerprint_metric = self.compute_fingerprint_metric(batch['fingerprint'])

        fusion_metric_fingerprint = 0.1 * image_metric + 0.1 * nmr_metric + 0.1 * smiles_metric + 0.7 * fingerprint_metric

        return fusion_metric_fingerprint
    
    def compute_fusion_metric_average(self, batch):
        nmr_metric = self.compute_nmr_metric(batch['nmr'])
        image_metric = self.compute_image_metric(batch['image'])
        smiles_metric = self.compute_smiles_metric(batch['smiles'])
        fingerprint_metric = self.compute_fingerprint_metric(batch['fingerprint'])

        fusion_metric_fingerprint = 0.25 * image_metric + 0.25 * nmr_metric + 0.25 * smiles_metric + 0.25* fingerprint_metric

        return fusion_metric_fingerprint

    def genGraphMetric(self, batch):
        with torch.no_grad():
            if self.graphMetric_method == 'image':
                metric = self.compute_image_metric(batch['image'])
            elif self.graphMetric_method == 'nmr':
                metric = self.compute_nmr_metric(batch['nmr'])
            elif self.graphMetric_method == 'smiles':
                metric = self.compute_smiles_metric(batch['smiles'])
            elif self.graphMetric_method == 'fingerprint':
                metric = self.compute_fingerprint_metric(batch['fingerprint'])
            elif self.graphMetric_method == 'fusion':
                metric = self.compute_fusion_metric(batch)
            elif self.graphMetric_method == 'fusion_image':
                metric = self.compute_fusion_metric_image(batch)
            elif self.graphMetric_method == 'fusion_nmr':
                metric = self.compute_fusion_metric_nmr(batch)
            elif self.graphMetric_method == 'fusion_smiles':
                metric = self.compute_fusion_metric_smiles(batch)
            elif self.graphMetric_method == 'fusion_fingerprint':
                metric = self.compute_fusion_metric_fingerprint(batch)
            elif self.graphMetric_method == 'fusion_average':
                metric = self.compute_fusion_metric_average(batch)
            else:
                raise ValueError(f"Unsupported graphMetric_method: {self.graphMetric_method}")

        return metric

