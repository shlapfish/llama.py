import cmd
from contextlib import contextmanager
from typing import Iterator

from llama import Model, Context, Mirostatv2Sampler, initialize_backend
from src.bindings import load_libllama
from src.context import Sequence

load_libllama("../llama.cpp")
initialize_backend(numa=False)


class Story:
    def __init__(self):
        self.model = Model("models/nous-hermes-llama2-13b.Q4_0.gguf", n_gpu_layers=1000)
        self.context = Context(self.model)
        self.sampler = Mirostatv2Sampler(self.context)
        self.sequence = Sequence(self.context)
        self.clear()

    def insert_text(self, text: str):
        self.last_logits = self.sequence.insert(text)
        self.context.process()

    def remove_last_paragraph(self):
        toks = self.sequence.tokens
        if toks[-1] == self.model.token_newline:
            toks.pop()
        if self.model.token_newline in toks:
            amount_to_truncate = list(reversed(toks)).index(self.model.token_newline)
        else:
            amount_to_truncate = len(toks) - 1
        self.sequence.truncate_end(amount_to_truncate)

    def write_paragraph(self, max_len=128) -> Iterator[str]:
        """
        Generates a paragraph (excluding the newline).
        :param max_len: The maximum length of the paragraph in tokens.
        """
        for _ in range(max_len):
            new_token = self.sampler.sample(self.last_logits)
            yield self.model.detokenize(new_token)
            if new_token == self.model.token_newline:
                return
            self.last_logits = self.sequence.insert(new_token)
            self.context.process()

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
        for _ in range(num):
            print(f"{len(self.current_options) + 1}. {prefix}", end="")
            par = prefix
            print(prefix, end="")
            with story.rollback():
                for piece in story.write_paragraph():
                    print(piece, end="")
                    par += piece
            self.current_options.append(par)
            print()

    def do_extend(self, arg: str):
        'extend [num] text: extend given text, generating num possible paragraphs.'
        num, text = arg.split(maxsplit=1)
        text = text.strip() + " "
        with story.rollback():
            story.insert_text(text)
            self.do_par(num, prefix=text)

    def do_pick(self, arg):
        'pick [n]: pick the n-th option and write it into the story'
        self.do_write(self.current_options[int(arg) - 1])

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
        print(story.sequence)

    def do_exit(self, arg):
        exit()


WriterShell().cmdloop()
