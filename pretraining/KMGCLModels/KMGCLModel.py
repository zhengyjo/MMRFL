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
        self.sign_predictor = nn.Linear(config.projection_dim[0]*2, 3)
        self.graphMetric_method = config.graphMetric_method
        self.nodeMetric_method = config.nodeMetric_method
        self.alpha = config.alpha
        self.device = config.device
        self.config = config

    def forward(self, batch):

        # GraphEmbedding & NodeEmbedding
        # Zhengyang's modification to use Molecule Dataset
        graph = batch['smiles_input'].batch_graph()
        #print(len(graph))
        def eye_like(tensor):
            return torch.eye(*tensor.size(), out=torch.empty_like(tensor))
        
        mask = torch.nonzero(batch['graph'].x[:, 0] == 5.0).squeeze(1)
        
        
        graphEmbedding, nodeEmbedding = self.graph_encoder(graph)
   
        nodeEmbedding = nodeEmbedding[mask] # Only extract the Carbon information
        graphEmbedding = F.normalize(graphEmbedding, p=2, dim=1)
        nodeEmbedding = F.normalize(nodeEmbedding, p=2, dim=1)

        # Node embedding processing for sign prediction
        # Get the number of rows in the embedding matrix
        num_rows = nodeEmbedding.size(0)

        # Initialize a list to store pairwise concatenated embeddings
        pairwise_embeddings = []

        # Generate pairwise concatenated embeddings
        for i in range(num_rows):
            for j in range(num_rows):  # Ensure pairs are unique
                # Concatenate embeddings from row i and row j
                concatenated = torch.cat((nodeEmbedding[i], nodeEmbedding[j]), dim=0)
                pairwise_embeddings.append(concatenated)

        # Convert list of tensors to a single tensor
        pairwise_embeddings_matrix = torch.stack(pairwise_embeddings)

        # Sign prediction
        # Apply the linear transformation
        # logits_sign = self.sign_predictor(pairwise_embeddings_matrix)

        # Define the cross-entropy loss function
        criterion = nn.CrossEntropyLoss()
        # nodeMetric
        
        
        ppm_diff = batch['peak']

        #print("ppm_diff shape")
        #print(ppm_diff.shape)
        if "CL" in self.nodeMetric_method:
            similarity_matrix_node = torch.matmul(nodeEmbedding, nodeEmbedding.T)
            positives_node = torch.diagonal(similarity_matrix_node)
            nodeLogits = similarity_matrix_node      # Scale similarities
            nodeLabels = eye_like(similarity_matrix_node)  # Identity labels
            nodeLoss = F.cross_entropy(nodeLogits, nodeLabels)
        elif "TL" in self.nodeMetric_method:
            similarity_matrix_node = torch.matmul(nodeEmbedding, nodeEmbedding.T)
            positives_node = torch.diagonal(similarity_matrix_node)
            mask_node = eye_like(similarity_matrix_node)  # Identity mask
            negatives_node = similarity_matrix_node - mask_node * 1e6  # Ignore diagonal
            hardest_negatives_node = negatives_node.max(dim=1)[0] 
            nodeLoss = F.relu(1 - positives_node + 0.3 - (1 - hardest_negatives_node)).mean() # Margin should be considered
        else:
            nodeMetric = F.softmax(ppm_diff, dim=-1)
            # nodeLoss
            nodeLogits = nodeEmbedding @ nodeEmbedding.T
            nodeLoss = F.cross_entropy(nodeLogits, nodeMetric) + F.cross_entropy(nodeLogits.T, nodeMetric.T)

        # graphMetric
        if "CL" in self.graphMetric_method:
            target_embedding = self.genTargetEmbedding(batch)
            target_embedding = F.normalize(target_embedding, p=2, dim=1)
            projection = Projection(target_embedding.size(-1), self.config.projection_dim, self.config.dropout).to(self.device)
            target_embedding = projection(target_embedding)
            similarity_matrix = torch.matmul(graphEmbedding, target_embedding.T)
            positives = torch.diagonal(similarity_matrix)
            graphLogits = similarity_matrix      # Scale similarities
            graphLabels = eye_like(similarity_matrix)  # Identity labels
            graphLoss = F.cross_entropy(graphLogits, graphLabels)
            
        elif "TL" in self.graphMetric_method:
            target_embedding = self.genTargetEmbedding(batch)
            target_embedding = F.normalize(target_embedding, p=2, dim=1)
            projection = Projection(target_embedding.size(-1), self.config.projection_dim, self.config.dropout).to(self.device)
            target_embedding = projection(target_embedding)
            similarity_matrix = torch.matmul(graphEmbedding, target_embedding.T)
            positives = torch.diagonal(similarity_matrix)
            graphLogits = similarity_matrix 
            mask = eye_like(similarity_matrix)  # Identity mask
            negatives = similarity_matrix - mask * 1e6  # Ignore diagonal
            hardest_negatives = negatives.max(dim=1)[0] 
            graphLoss = F.relu(1 - positives + 0.3 - (1 - hardest_negatives)).mean() # Margin should be considered
        else:
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
    
    def compute_image_embedding(self,image):
        image_embeddings = self.image_encoder(image)
#         projection = Projection(image_embeddings.size(-1), self.config.projection_dim, self.config.dropout).to(self.device)
#         image_embeddings = projection(image_embeddings)
        return image_embeddingss
    
    def compute_nmr_embedding(self,nmr):
        nmr_embeddings = self.nmr_encoder(nmr)
#         projection = Projection(nmr_embeddings.size(-1), self.config.projection_dim, self.config.dropout).to(self.device)
#         nmr_embeddings = projection(nmr_embeddings)
        return nmr_embeddings
    
    def compute_smiles_embedding(self,smiles):
        smiles_embeddings = self.smiles_encoder(smiles)
#         projection = Projection(smiles_embeddings.size(-1), self.config.projection_dim, self.config.dropout).to(self.device)
#         smiles_embeddings = projection(smiles_embeddings)
        return smiles_embeddings
    
    def compute_fingerprint_embedding(self,fingerprint):
        fingerprint_embeddings = fingerprint
#         projection = Projection(fingerprint_embeddings.size(-1), self.config.projection_dim, self.config.dropout).to(self.device)
#         fingerprint_embeddings = projection(fingerprint_embeddings)
        return fingerprint_embeddings
    
    
    def genTargetEmbedding(self, batch):
        with torch.no_grad():
            if self.graphMetric_method == 'image_CL' or self.graphMetric_method == 'image_TL':
                targetEmbedding = self.compute_image_embedding(batch['image'])
            elif self.graphMetric_method == 'nmr_CL' or self.graphMetric_method == 'nmr_TL':
                targetEmbedding = self.compute_nmr_embedding(batch['nmr'])
            elif self.graphMetric_method == 'smiles_CL' or self.graphMetric_method == 'smiles_TL':
                targetEmbedding = self.compute_smiles_embedding(batch['smiles'])
            elif self.graphMetric_method == 'fingerprint_CL' or self.graphMetric_method == 'fingerprint_TL':
                targetEmbedding = self.compute_fingerprint_embedding(batch['fingerprint'])
        return targetEmbedding
            
        

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

