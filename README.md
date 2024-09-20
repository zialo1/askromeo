# askromeo
Python Programm. A Prompt/Interface to Pytorch LLM Model

## What it is

This programm is a Prompt or interface for torch Large Language Models. The model given here is trained on Shakespeares language. The programm loads this model, then askromeo asks's the user for his words which are rephrased and processed as if they were question in a Shakespeare Text. From there on the LLM inventes how the book could continue.
The LLM inventes words and sentences to this question or prompt.
Use '*' for help when the program is running.

## References
LLM is based on exercise in course Contemporary Machine Learning, University of Basel, 2024. It contains code from Andrei Karpathy's Youtube course. The model was trained on Data from the Project Gutenberg.

## Legal
No responsability for the output is assumed.

## Install and Usage

### Install
install python
install torch, numpy
### Save the model
save the file 'model_statedict_40914' in the same directory as the script.

### First-Run
run "python3 askromeo.py"

### Help
press '*' while the program runs.

### Further Comments
#### Usage with an existing model
the class romeoPrompt needs to initialized with the model as parameter. The functions encoder, the decoder and a list of chars of the input are also needed for the initialization of the class.
You can use the interface in REPL mode or replace this line in the section main.
In REPL mode use the interface functions: processcmd , mode2askromeo or mode1prepare with strings as arguments.
