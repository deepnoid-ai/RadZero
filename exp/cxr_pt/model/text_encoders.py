import open_clip
import torch
from transformers import AutoModel
from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.siglip.modeling_siglip import SiglipTextModel


def build_text_encoder(config):
    if config.model_type == "siglip_text_model":
        model = SiglipTextModel.from_pretrained(config.pretrained_name_or_path)
    elif config.model_type == "clip_text_model":
        model = CLIPTextModel.from_pretrained(config.pretrained_name_or_path)
    elif config.model_type == "mpnet":
        model = AutoModel.from_pretrained(config.pretrained_name_or_path)
    elif config.model_type == "biomedclip":
        # reference: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/main/biomed_clip_example.ipynb
        clip_model, preprocess = open_clip.create_model_from_pretrained(
            config.pretrained_name_or_path
        )
        model = clip_model.text
    elif config.model_type == "bioclinicalmpbert":
        model = AutoModel.from_pretrained(
            config.pretrained_name_or_path, output_hidden_states=True
        )
    else:
        raise NotImplementedError()

    return model


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def aggregate_tokens(embeddings, caption_ids, idxtoword):

    batch_size, num_layers, num_words, dim = embeddings.shape
    embeddings = embeddings.permute(0, 2, 1, 3)
    agg_embs_batch = []
    sentences = []

    # loop over batch
    for embs, caption_id in zip(embeddings, caption_ids):

        agg_embs = []
        token_bank = []
        words = []
        word_bank = []

        # loop over sentence
        for word_emb, word_id in zip(embs, caption_id):

            word = idxtoword[word_id.item()]

            if word == "[SEP]":
                new_emb = torch.stack(token_bank)
                new_emb = new_emb.sum(axis=0)
                agg_embs.append(new_emb)
                words.append("".join(word_bank))

                agg_embs.append(word_emb)
                words.append(word)
                break

            if not word.startswith("##"):
                if len(word_bank) == 0:
                    token_bank.append(word_emb)
                    word_bank.append(word)
                else:
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))

                    token_bank = [word_emb]
                    word_bank = [word]
            else:
                if word.startswith("##"):
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])

        agg_embs = torch.stack(agg_embs)
        padding_size = num_words - len(agg_embs)
        paddings = torch.zeros(padding_size, num_layers, dim)
        paddings = paddings.to(agg_embs.device)
        words = words + ["[PAD]"] * padding_size

        agg_embs_batch.append(torch.cat([agg_embs, paddings]))
        sentences.append(words)

    agg_embs_batch = torch.stack(agg_embs_batch)
    agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
    return agg_embs_batch, sentences
