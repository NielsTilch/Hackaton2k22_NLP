from transformers import WEIGHTS_NAME, CONFIG_NAME
import os,torch
from transformers import AutoTokenizer
import torch.utils.data as t_data
import Dataset_and_train_functions as dtf
import readconllfile as rcf
import torch.optim as optim
from MultiLabelsTokenClassification import BertForTokenMul

lr = 1e-5
wd = 1e-4
epoches = 15
tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert-base-cased")
Model = BertForTokenMul.from_pretrained("jjzha/jobbert-base-cased", num_labels=5)
dossierchemin = "./data/Skills/"
fichier = "skillspan_tech_train.conll"
adress = dossierchemin
df = rcf.read_all_conll_file(adress, "train")
df_len = len(df['Seqs'])
datas = dtf.SequencesData(df, tokenizer)
dl = t_data.DataLoader(datas,batch_size=16, shuffle = True)
optimizer = optim.Adam(Model.parameters(), lr = lr, weight_decay=wd, amsgrad = True)
dtf.train(tokenizer, Model, dl, optimizer, epochs = epoches, num_sample = df_len)

fichier_v = "skillspan_tech_dev.conll"
adress_v = dossierchemin
df_v = rcf.read_all_conll_file(adress_v,"dev")
df_v_len = len(df_v['Seqs'])
datas_v = dtf.SequencesData(df_v, tokenizer,flag=1)
dl_v = t_data.DataLoader(datas_v, batch_size = 16, shuffle = False)
dtf.validation(tokenizer, Model, dl_v, df_v_len)

output_dir = "./models/"
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

torch.save(Model.state_dict(), output_model_file)
Model.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(output_dir)