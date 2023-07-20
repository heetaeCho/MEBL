from transformers import BertTokenizer, BertModel
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import logging
import torch

logging.set_verbosity_error()

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# device = 'cpu'

class EmbeddingModel:
    def __init__(self):
        print("====== Embedding Models Loading Start ======")
        self.Bert = \
            (BertTokenizer.from_pretrained('bert-base-uncased'), BertModel.from_pretrained('bert-base-uncased').to(device))
        self.CodeBert = \
            (AutoTokenizer.from_pretrained('microsoft/codebert-base'), AutoModel.from_pretrained('microsoft/codebert-base').to(device))
        self.NatGen = \
            (AutoTokenizer.from_pretrained("saikatc/NatGen"), AutoModelForSeq2SeqLM.from_pretrained("saikatc/NatGen").to(device))
        print("======= Embedding Models Loading End =======")

    def embedding(self, channel, data):
        if channel == 'nl':
            return self._nl_channel(data)
        elif channel == 'sc_nl':
            return self._sc_nl_channel(data)
        elif channel == 'sc_pl':
            return self._sc_pl_channel(data)
        else:
            raise "Check the embeeding channel"

    def _nl_channel(self, data):
        with torch.no_grad():
            encode_input = self.Bert[0](data, return_tensors='pt', padding="max_length", truncation=True).to(device)
            if encode_input['input_ids'].shape[-1] > 512:
                for key in encode_input.keys():
                    encode_input[key] = torch.cat((encode_input[key][:,:511], encode_input[key][:,-1].view(1,-1)), dim=1)
            output = self.Bert[1](**encode_input)
            return output[0]

    def _sc_nl_channel(self, data):
        with torch.no_grad():
            encode_input = self.CodeBert[0].tokenize(data, max_length=510, padding="max_length", truncation=True)
            encode_input = [self.CodeBert[0].cls_token] + encode_input + [self.CodeBert[0].eos_token]
            encode_input = self.CodeBert[0].convert_tokens_to_ids(encode_input)
            return self.CodeBert[1](torch.tensor(encode_input)[None,:].to(device))[0]

    def _sc_pl_channel(self, data):
        with torch.no_grad():
            encode_input = self.NatGen[0](data, return_tensors='pt', padding="max_length", truncation=True).to(device)
            output = self.NatGen[1].encoder(
                input_ids = encode_input['input_ids'][:, :512],
                attention_mask = encode_input['attention_mask'][:, :512],
                return_dict=True
            )
            return output[0]
        
# if __name__ == "__main__":
#     em = EmbeddingModel()
#     print(em.CodeBert)