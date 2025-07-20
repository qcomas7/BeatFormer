import torch


def orthonormal_loss(att_blocks):
    patches = att_blocks.shape[-1] // 3
    att_blocks = att_blocks.reshape(att_blocks.shape[0], att_blocks.shape[1], att_blocks.shape[2], att_blocks.shape[3],
                                    patches, 3, patches, 3).permute(0, 1, 2, 3, 4, 6, 5, 7)
    att_blocks = att_blocks.reshape(att_blocks.shape[0], att_blocks.shape[1], att_blocks.shape[2], att_blocks.shape[3],
                                    -1, 3, 3)
    batch, blocks, attention, qkv, patches, x, y = att_blocks.shape
    reshaped_att = att_blocks.view(batch * blocks * attention * qkv * patches, 3, 3)
    dot12 = torch.matmul(reshaped_att[:, 0, :].unsqueeze(1), reshaped_att[:, 1, :].unsqueeze(-1))
    dot13 = torch.matmul(reshaped_att[:, 0, :].unsqueeze(1), reshaped_att[:, 2, :].unsqueeze(-1))
    dot23 = torch.matmul(reshaped_att[:, 1, :].unsqueeze(1), reshaped_att[:, 2, :].unsqueeze(-1))
    orthogonality_loss = torch.sum(dot12 ** 2) + torch.sum(dot13 ** 2) + torch.sum(dot23 ** 2)
    row_norms = torch.norm(reshaped_att, dim=-1)
    unit_loss = torch.sum((row_norms - 1) ** 2)
    total_loss = orthogonality_loss + unit_loss
    return total_loss / reshaped_att.shape[0]











'''
import torch


def orthonormal_loss(att_blocks):
    att_blocks = att_blocks.reshape(att_blocks.shape[0], att_blocks.shape[1], att_blocks.shape[2], -1, 3, 3)
    blocks, attention, qkv, patches, x, y = att_blocks.shape
    reshaped_att = att_blocks.view(blocks * attention * qkv * patches, 3, 3)
    dot12 = torch.matmul(reshaped_att[:, 0, :].unsqueeze(1), reshaped_att[:, 1, :].unsqueeze(-1))
    dot13 = torch.matmul(reshaped_att[:, 0, :].unsqueeze(1), reshaped_att[:, 2, :].unsqueeze(-1))
    dot23 = torch.matmul(reshaped_att[:, 1, :].unsqueeze(1), reshaped_att[:, 2, :].unsqueeze(-1))
    orthogonality_loss = torch.sum(dot12 ** 2) + torch.sum(dot13 ** 2) + torch.sum(dot23 ** 2)
    row_norms = torch.norm(reshaped_att, dim=-1)
    unit_loss = torch.sum((row_norms - 1) ** 2)
    total_loss = orthogonality_loss + unit_loss
    return total_loss / reshaped_att.shape[0]


'''