#!python3
# askromeo.py
# LLM that inventes Shakespearian word salad
# (C) 2024: A. Hanselmann Part User Interaction and modell access
# based on 2024, Machine Learning Course for Physicist Unibas (CH)
# based on Andreji Karpathys Code and his Course "Let's build GPT: from scratch, in code, spelled out"
# version 0.21, 2024-09
#
# ------------------------------------------
# howto run: pip install torch, pip
# run python3 askromeo.py
# the file 'model_stat_dict_0240914' is neeeded

# 2do: Why? Question doesn't work. Enter doesn't work. Free Prompting doesn't work.

# ---------------------------------
# standard imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# replace this with another model if given
# was created with 'torch.save(model.state_dict(),"model_state_dict")'
fn_of_model = 'model_statedict_40914'

torch.seed()

# do not change this part: PART_MODEL_USED_AND_NEEDED
#  -----------------------------------
# hyperparameters - used for import. The model needs to be trained with
# these and also the following classes contained here
batch_size = 24  # how many independent sequences will we process in parallel?
block_size = 48  # what is the maximum context length for predictions?
max_iters = 50000
eval_interval = 100
learning_rate = 9e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 72
n_head = 8
n_layer = 4
dropout = 0.0
# ------------

chars = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3',
         ':', ';', '?',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
         'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
         'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
         'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w', 'x', 'y', 'z']

vocab_size = len(chars)

# mapping functions
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l1): return ''.join([itos[i] for i in l1])


class Embedder(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T,C)
        return tok_emb + pos_emb  # (B,T,C)


class FeedFoward(nn.Module):
    """ a.k.a. Multi-Layer-Perceptron: a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # will gewichte nicht ändern, keine gradienten und so, nutzen register_buffer
        # self.trill
        # more efficient for constants
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # dimension of k: (Batch, Token, Channel) = (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        # alphascalarprodukt, normieren mit C embedding_dimension
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        # print(q.shape)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([Head(head_size)
                                   for n in range(0, num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([self.heads[i](x)
                        for i in range(self.num_heads)], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        """
        init a MultiHeadAttention-Layer and a FeedForward-Layer, and two LayerNorm-Layers
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        # standard transormer architexture - best practice
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """implement forward pass:
        first, a ResNetBlock with f(x) = MultiHeadAttention(LayerNorm1(x))
        second, a ResNetBlock with f(x) = FeedForward(LayerNorm2(x))
        third: return result
        """

        # resNet bedeutet identität
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.embedder = Embedder(vocab_size, n_embd, block_size)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.final_layernorm = nn.LayerNorm(n_embd)  # final layer norm
        self.language_model_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        '''
        map idx to logits.
        '''
        # logits =
        '''
        keep loss calculation and generate below untouched
        '''

        logits = self.embedder(idx)
        logits = self.blocks(logits)
        logits = self.final_layernorm(logits)
        logits = self.language_model_head(logits)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            E = -torch.log(probs)
            Z = torch.sum(torch.exp(-E/temperature))
            probs = torch.exp(-E / temperature) / Z

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# end of PART_MODEL_USED_AND_NEEDED
# -----------------------------------------------

# Initialize Model and Load from Disk

# run the user interpreter

class randomList:
    def __init__(self, content, seperator='\n'):
        if isinstance(content, list):
            self.listelements = content
        else:
            self.listelements = [
                ele for ele in content.split(seperator) if ele != ""]

        self.length = len(self.listelements)

    def __repr__(self):  # doesn't work for variables
        return self.shuffle()

    def shuffle(self):
        randomi = int(torch.randint(self.length, (1,)))
        return self.listelements[randomi]


class romeoPrompt:
    '''a class for prompting a llm using structure of shakespeaere plays to trigger answers.
    autor: alex hanselmann'''

    # messages and informations
    welcomemsg = '''
Prompt Interfact for Large Language Model (loaded). 
The model is trained on Shakespeares language. The user can enter words that are rephrased and processed as if they were question in a Shakespeare Text. 
The LLM inventes words and sentences to this question or prompt.
No responsability for the output is assumed.
Use '*' for help.
Try: 'Multi-layer perceptrons are' or 'I am hungry.'.
Try: '%.2' and press Enter. Use '.' to quit.
'''
    helptip = " [help with '*']"
    helpmsg = '''
Some Help.
Available Commands:
'_'    : Display temperature and maximal length of an answer.
'?'    : A pregiven question from Shakespeares book is choosen.
'%+'   : Temperature is slightly increased, e.g. '%1.1' works too.
'%-'   : Temperature is slightly decreased.
'=400' : Generates a free text of 400 Characters. Any Number goes.
         This sets also the number of tokens for other queries.
'!'    : Resets temperature and length of answers to default.

When your question ends with '?', it's rephrased so that Romeo answers.
Either MENEIUS, GLOUCESTER or DUKE VINCENTIO ask him in your place.
Tips: When you press Enter, the last input is reused.
Try 'Ay! Can't you help?' or '=30' for short answers and enter '?'

LLM is based on exercise in course Contemporary Machine Learning, University of Basel, 2024. It contains code from Andrei Karpathy's Youtube course. 
No responsability for the output is assumed.The model was trained on Data from the Project Gutenberg.
'''
    # randomLists
    # they gonna ask 'ROMEO:'
    poi = randomList(['MENENIUS:', 'GLOUCESTER:', 'DUKE VINCENTIO:'])

    shakespearequestions = randomList('''
What is the time o' the day?
What is the matter?
How now! what news?
Pray, sir, in what?
But, hark, what noise?
And is this all?
Who's here?
You understand me?
What's that, I pray?
Why, what's a moveable?
By any likelihood he show'd to-day?
Never?
Good news or bad, that thou comest in so bluntly?
Is thy news good, or bad? answer to that;
So. What trade are you of, sir?
O, ay, what else?
Say, is my kingdom lost?
Welcome, my lord what is the news?
What sayest thou? speak suddenly; be brief.
Who saw the sun to-day?
How now! what news?
Would not infect his reason?
Will you take eggs for money?
She will be pleased; then wherefore should I doubt?
And then?
And what of all this?
Where is he?
What's this?
Art thou not Romeo and a Montague?
Your eyes do menace me: why look you pale?
Art thou sure of this?
And what to?
And bring thy news so late?
Alas, why would you heap these cares on me?
'''
                                      )

    # give pytorch-model, the encode and decoder for tokens and the charset. the last is needed to validate input
    def __init__(self, model, encoder, decoder, inputchars):
        ''' initializes interface to pytorch model. inputchars enable
validation of input. if model is trained no further data is needed,
else state_dict  must be loaded with loadmodel. check torch.save()
'''
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.temperature = 0.92
        self.length_of_answer = 70
        self.validate_set = set(inputchars)
        self.context = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.lastquestion = None

    # loadmodel: filename with saved state_dict of torch_model needed if model is untrained.
    def loadmodel(self, fn):
        self.model.load_state_dict(torch.load(fn_of_model, weights_only=True))
        return "model is loaded"

    # interpretation of input that is a command: called by main
    def processcmd(self, instring):

        match instring:
            case '.':  # quit
                print("Bye!")
                quit()

            case '!':  # reset
                self.temperature = 1
                self.length_of_answer = 70
                return "Reseting maxlength and temperature."+self.helptip

            case '_':  # display parameters
                return f'Temperature {self.temperature:.03f}, maximal generated tokens={self.length_of_answer}'+self.helptip

            case '?':  # call generate_answer with a prepared  random question
                randomquestion = self.shakespearequestions.shuffle()
                return self.generate_result(self.mode1prepare(randomquestion))

            case '=':  # freerun with pregiven number
                return self.freerun()

            case '*' | '%': # help or wrong usage
                return self.help()

        # mode change temperature, produces error if string empty
        if instring[0] == '%':
            if "+" in instring:
                self.temperature += 0.05 * instring.count('+')
            elif "-" in instring:
                self.temperature -= 0.05 * instring.count('-')
            else:
                self.temperature = float(instring[1:])
            return f'Temperature is now {self.temperature:.03f}'

        # change length of output
        elif instring[0] == '=':
            if instring[1:].isdigit():
                number = int(instring[1:])
                self.length_of_answer = number
                # generate free text
                return self.freerun()
            else:
                return self.helptip

        return ""  # not a command, process

    # called by main to check input
    def validate_input(self, input):
        if self.validate_set.issuperset(input):
            return ""  # ok
        else:
            return "Input contains characters that are not in text. " +\
                   "Try again." + self.helptip

    # root method: calls the model and generate tokens
    def applymodel(self, context, numtokens=70):
        try:  # Error if propapbility too low
            return decode(self.model.generate(context,
                          max_new_tokens=numtokens,
                          temperature=self.temperature)[0].tolist())
        except RuntimeError:  # try
            # warning
            print(
                "Temperature/Probability too low." +
                "Set Temperature to around 1. Type: 'T1' or 'T.5'")
            raise UserWarning

    # take pre-tokens, produce new tokens following them
    def generate_result(self, predata=None, nodotcheck=False):
        if predata is not None:
            newcontext = predata+self.context
            len_predata = len(predata)
        else:
            newcontext = self.context
            len_predata = 0

        prefix = "\n"  # a kind of flag that is activated in 2nd looü
        while True:
            # get an answer
            try:  # can raise a Warning if temp <0.05
                answer = self.applymodel(
                    newcontext, numtokens=self.length_of_answer)
            except UserWarning:
                self.temperature = 0.5
                break

            # Unless ? we don't want CHANGE OF SPEAKER (has ':\n')
            if '?' not in answer and ':\n' in answer:
                continue

            # look for sentences, we only take such anwers

            pos = answer.rfind('.', len_predata+1)
            if pos > 0 or nodotcheck:  # sentence was returned. break inner while loop
                return f'[{pos-len_predata}]*{prefix}{answer[:pos+1]} \n'
            elif self.temperature < 0.4:
                return f'*{prefix}{answer}...'

            print('.', end='')  # direct output
            prefix = "\n"

    # get a question, return prefix-tokens for generate_answer, called from main
    def mode2askromeo(self, aquestion):
        ''' takes question, produces prefix-tokens for generate_answer()  '''
        # "Why bother? -> "MENEIUS: Why bother? ROMEO: "
        question = aquestion[0].upper() + aquestion[1:]

        question_to_romeo = '\n'.join(
            ('', self.poi.shuffle(), question, '', 'ROMEO:',''))
        self.lastquestion = question_to_romeo
        return torch.tensor(self.encoder(question_to_romeo))

    # get normal text, return tokens ready for generate
    def mode1prepare(self, text,rerun=False):
        ''' takes normal text, producs prefix-tokens for generate_answer() '''
        instring = text[0].upper()+text[1:]
        self.lastquestion = instring
        # Make it the beginning of a sentence, trigger a continuatio
        if rerun:
            postfix=""
        else:
            postfix=" "
        return torch.tensor(self.encoder(instring + postfix))

    # generates tokens, number is in length_of_answer
    def freerun(self):
        if self.firstrun():
            self.lastquestion = "" # if user asked a question before it isn't overwritten
        return self.applymodel(self.context, numtokens=self.length_of_answer)

    # prints help message
    def help(self):
        return self.helpmsg

    # check if the interface can rerun an old command
    # lastquestion can be "" if freerun() was used. this is the only case.
    def firstrun(self):
        if self.lastquestion is None:
            return True
        return False

    def __repr__(self):
        return self.welcomemsg


# MAIN
# ----------------------------------
if __name__ == '__main__':
    model = TransformerModel()
    # set model as first parameter or use method loadmodel
    interface = romeoPrompt(model, encode, decode, chars)
    interface.loadmodel(fn_of_model)
    print(interface)  # welcomemessage

    try:
        while True:  # quit loop by enter '.' as said in welcomemessage
            # ask for an input that is extended by the llm
            # until it's a closed sentence
            instring = input("Prompt> ")

            if instring == "":
                if interface.firstrun():
                    print(interface.help())
                else:  # redo with the same predata
                    if interface.lastquestion == "":  # last: free run, do again
                        print(interface.freerun())
                    else:
                        print(interface.generate_result(
                            interface.mode1prepare(interface.lastquestion, rerun=True)))
                continue

            # cmd processing. 2 cmd produce output (=?) others errtxt
            res = interface.processcmd(instring)
            if res != "":
                print(res)
                continue

            res = interface.validate_input(instring)
            if res != "":
                print(res)
                continue

            # question mode: rephrase ans let ROMEO answer
            if instring[-1] == '?':
                predata = interface.mode2askromeo(instring)
            # normal mode: just use input as prefix
            else:
                predata = interface.mode1prepare(instring)

            print(interface.generate_result(predata))

    except KeyError:  # where is the KeyError?
        pass
