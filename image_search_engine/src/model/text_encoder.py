from torch import nn
from image_search_engine.src import config as CFG
from transformers import DistilBertConfig, DistilBertModel

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.TEXT_ENCODER_MODEL, pretrained=CFG.PRETRAINED, trainable=CFG.TRAINABLE):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]