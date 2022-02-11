import numpy as np
import torch
from tqdm import tqdm


def omission(examples, trainer, tokenizer, model) -> list:
    """
    examples: torch.utils.data.Dataset
    trainer: transformers.Trainer
    tokenizer: transformers.AutoTokenizer
    model: transformers.AutoModelForSequenceClassification

    return: a list of the effects of removing each token from inputs in examples
    """
    input_ids_arr = np.array([np.array(e['input_ids']) for e in examples])
    token_type_arr = np.array([np.array(e['token_type_ids']) for e in examples])
    attention_arr = np.array([np.array(e['attention_mask']) for e in examples])

    # We just consider those validation samples which are correctly classified by the main model
    pred = trainer.predict(examples) # note to use main_trainer not other trainers like tiny_trainer
    ok_list = []
    for i in range(len(pred[0])):
        if np.argmax(pred[0][i]) == pred[1][i]:
            ok_list.append(i)

    res = []
    for pos in tqdm(ok_list, position=0, leave=False):
        pre = pred[0][pos]
        input_ids = torch.tensor([input_ids_arr[pos]], dtype=torch.long).to('cuda')
        tokens = tokenizer.convert_ids_to_tokens(input_ids.view(-1).cpu().numpy())

        wh = np.where(attention_arr[pos]==0)[0]
        if len(wh) == 0:
            border = len(attention_arr[pos])
        else:
            border = wh[0]
        for i in range(1, border-1):
            input_ids = torch.tensor([np.delete(input_ids_arr[pos], i)], dtype=torch.long).to('cuda')
            token_type_ids = torch.tensor([np.delete(token_type_arr[pos], i)], dtype=torch.long).to('cuda')
            attention_ids = torch.tensor([np.delete(attention_arr[pos], i)], dtype=torch.long).to('cuda')            

            A = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_ids)
            res.append((pos, tokens[i], pre - A.logits[0].detach().cpu().numpy()))

    return res
    