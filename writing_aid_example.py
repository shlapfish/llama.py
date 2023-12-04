import cmd
from contextlib import contextmanager
from copy import copy
from types import SimpleNamespace

from llama import Model, Context, Mirostatv2Sampler
from bindings import load_libllama, lib
from llama.context import Sequence

load_libllama("../llama.cpp")


class Story:
    def __init__(self):
        self.model = Model("models/nous-hermes-llama2-13b.Q5_K_M.gguf", n_gpu_layers=1000)
        self.context = Context(self.model, context_size=10000, rope_scaling_type=lib.LLAMA_ROPE_SCALING_YARN, rope_freq_scale=1.)
        self.sampler = Mirostatv2Sampler(self.context, target_entropy=2.)
        self.sequence = Sequence(self.context)
        self.last_logits = self.sequence.insert(self.model.token_bos)
        self.clear()

    def insert_text(self, text: str):
        self.last_logits = self.sequence.insert(text)
        self.context.process()

    def remove_last_paragraph(self):
        toks = self.sequence.tokens
        if toks[-1] == self.model.token_newline:
            toks.pop()
        if self.model.token_newline in toks:
            amount_to_truncate = 1 + list(reversed(toks)).index(self.model.token_newline)
            self.sequence.truncate_end(amount_to_truncate)
            self.insert_text("\n")
        else:
            self.clear()

    def generate_paragraphs(self, amount: int, max_len=512) -> list[str]:
        """
        Generates a paragraph (excluding the newline).
        :param amount: The amount of paragraphs to generate
        :param max_len: The maximum length of the paragraph in tokens.
        """
        batch_n = self.context.max_batch_size
        if amount > batch_n:
            return self.generate_paragraphs(batch_n) + self.generate_paragraphs(amount - batch_n)
        states = [SimpleNamespace(seq=copy(self.sequence),
                                  logits=copy(self.last_logits),
                                  tokens=[],
                                  sampler=copy(self.sampler))
                  for _ in range(amount)]

        for _ in range(max_len):
            for state in states:
                if not state.tokens or state.tokens[-1] != "\n":
                    new_tok = state.sampler.sample(state.logits)
                    if new_tok in (self.model.token_eos, self.model.token_newline):
                        state.tokens.append("\n")
                    else:
                        state.tokens.append(self.model.detokenize(new_tok))
                        state.logits = state.seq.insert(new_tok)
            self.context.process()
        for s in states:
            s.seq.clear()
        return ["".join(s.tokens).strip() for s in states]

    def clear(self):
        self.sequence.clear()
        # insert beginning-of-sequence token
        self.last_logits = self.sequence.insert(self.model.token_bos)
        self.context.process()

    @contextmanager
    def rollback(self):
        """For making hypothetical changes and easily rolling them back."""
        initial_length = len(self.sequence)
        initial_logits = self.last_logits
        try:
            yield None
        finally:
            self.last_logits = initial_logits
            self.sequence.truncate_end(len(self.sequence) - initial_length)


story = Story()


def parse_num(arg: str) -> int | None:
    try:
        return int(arg)
    except ValueError:
        print(f"Expected a number, got {arg}")


class WriterShell(cmd.Cmd):
    intro = "Welcome to adventure writer! Type help or ? to list commands.\n"
    prompt = "> "

    def __init__(self):
        super().__init__()
        self.current_options = []

    def do_clear(self, arg=""):
        story.clear()
        self.current_options.clear()

    def do_write(self, arg):
        'write [text]: insert a paragraph of text into the story'
        arg = arg.strip()
        story.insert_text(arg + "\n")
        print(f"{len(story.sequence)}/{story.context.context_size()}")
        self.current_options.clear()

    def do_save(self, arg):
        with open(arg, 'w') as file:
            file.write(str(story.sequence))

    def do_load(self, arg):
        self.do_clear()
        try:
            with open(arg, 'r') as file:
                for line in file.readlines():
                    print(line)
                    self.do_write(line.strip())
        except FileNotFoundError:
            print(f"File not found: {arg}")

    def do_par(self, arg, prefix=""):
        'par [num]: generate num possible paragraphs'
        num = parse_num(arg)
        if not num:
            return
        for option in story.generate_paragraphs(num):
            print(f"{len(self.current_options) + 1}. {prefix}{option}\n")
            self.current_options.append(option)

    def do_extend(self, arg: str):
        'extend [num] text: extend given text, generating num possible paragraphs.'
        num, text = arg.split(maxsplit=1)
        text = text.strip() + " "
        with story.rollback():
            story.insert_text(text)
            self.do_par(num, prefix=text)

    def do_pick(self, arg):
        'pick [n]: pick the n-th option and write it into the story'
        try:
            self.do_write(self.current_options[int(arg) - 1])
        except IndexError:
            print("No such option.")

    def do_undo(self, arg):
        'removes last paragraph'
        self.current_options.clear()
        story.remove_last_paragraph()
        print("removed last paragraph")

    def do_instruct(self, arg: str):
        'instruct [n] command: write n paragraphs following the command.'
        num, instruction = arg.split(maxsplit=1)
        with story.rollback():
            story.insert_text(f"### Instruction:\n{instruction.strip()}\n")
            self.do_par(num)

    def do_print(self, arg):
        print(str(story.sequence).replace("\n", "\n\n"))

    def do_exit(self, arg):
        exit()


WriterShell().cmdloop()
