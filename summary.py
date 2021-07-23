import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

#src_text = ["""I had a call with Steven day before yesterday, regarding download of new weekly and monthly reports. We need to decide next action and this meat. I will give you a quick context of what the business is, it might be useful for the team Lloyd's of London, operates with three key entities, insurance companies carriers or brokers cover holder, or mga.perfect so what does a broker do here.brokers bring in business careers covers the risk mg kind of acts like offshoring center acts like a third party wing, that takes care of courier. And they are the biggest insurance.So our client Lloyd has outsourced work for Hyperion group Hyperion is a combination of couple of brokers and mg. There is rk hedge, which is broker howden which is broker edge CL which is again an mg.Cool. As far as I understood about Lloyds is that you have syndicates, that is usually a group of people who come together to cover a particular type of risk, and under syndicate you will also be having agents who will work for syndicate or sometimes with the syndicates themselves, and manage their day to day business, and who will also manage their accountants and underwriters. So after agents, I think the next guys will be the brokers who are the Lloyds authorized brokers who will be approached for any kind of insurances m&a.Yes, these carriers, what I call as insurance company. They have been, they have to be in Lloyds to be called as a syndicate, and these syndicates will have their underwriters and these underwriters work with Lloyds registered brokers. You are understanding it right but only thing is that to be a syndicate an insurance company must be operated must be operated from Lloyds. It should be a syndicate rather than registered as company with Lloyds Lloyds cannot have companies registered under them. Our to do for this week is to research about Lloyd's London market and understand the risks covered. We will group up next week, guys."""]
def absSummary():
    modelname = 'google/pegasus-xsum'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(modelname)
    model = PegasusForConditionalGeneration.from_pretrained(modelname).to(torch_device)
    with open("minutesOM/SRH Hochschule Heidelberg 1Text.txt") as text_file:
        contents = text_file.read()
    #print(contents)

    batch = tokenizer.prepare_seq2seq_batch(contents,truncation = True,padding="longest",return_tensors='pt').to(torch_device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated,skip_special_tokens=True)
    
    return tgt_text[0]


if __name__=='__main__':
    absSummary()